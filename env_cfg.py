"""Builds the mjlab flat-terrain velocity env from [config.py](config.py).

`sesame_flat_env_cfg(play=False)` returns a `ManagerBasedRlEnvCfg` with every
toggle / weight / curriculum stage sourced from `config`. Other files read
from this module; nothing in here mutates `config`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg

import config as C
from sesame_robot import (
    ACTION_SCALE,
    FOOT_GEOM_NAMES,
    FOOT_SITE_NAMES,
    get_robot_cfg,
)


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Mjlab uses slightly different names than the ones in `config.REWARDS` for
# one term (`dof_pos_limits` <-> `joint_pos_limits`). Everything else matches.
_MJLAB_REWARD_NAME = {
    "track_linear_velocity":  "track_linear_velocity",
    "track_angular_velocity": "track_angular_velocity",
    "upright":                "upright",
    "action_rate_l2":         "action_rate_l2",
    "action_acc_l2":          "action_acc_l2",       # added below
    "dof_pos_limits":         "dof_pos_limits",
    "air_time":               "air_time",
    "foot_slip":              "foot_slip",
    "soft_landing":           "soft_landing",
    "posture":                "posture",             # added below
    "electrical_power":       "electrical_power",    # added below
}


# ---------------------------------------------------------------------------
# Command-range curriculum
# ---------------------------------------------------------------------------
class _CommandStage(TypedDict):
    step: int
    x_com: tuple[float, float]
    y_com: tuple[float, float]
    z_com: tuple[float, float]


def _is_forward_only(stage: _CommandStage) -> bool:
    """True iff a stage commands positive-x only (y and z pinned to zero)."""
    x_lo, _ = stage["x_com"]
    y_lo, y_hi = stage["y_com"]
    z_lo, z_hi = stage["z_com"]
    return x_lo >= 0.0 and y_lo == y_hi == 0.0 and z_lo == z_hi == 0.0


def _rel_forward_for(stage: _CommandStage) -> float:
    """Fraction of envs to be forward-only for this stage.

    mjlab samples `is_standing_env` and `is_forward_env` as independent
    Bernoullis, and standing wins on conflict. So the effective population is
      standing : REL_STANDING_ENVS
      forward  : (1 - REL_STANDING_ENVS) * rel_forward_envs
      normal   : (1 - REL_STANDING_ENVS) * (1 - rel_forward_envs)
    On forward-only stages we want to fill all non-standing mass with
    forward-clamped commands, so `rel_forward_envs = 1 - REL_STANDING_ENVS`.
    Once any axis opens, disable forward-clamping entirely.
    """
    return (1.0 - C.REL_STANDING_ENVS) if _is_forward_only(stage) else 0.0


class command_range_curriculum:
    """Stage-based expansion of UniformVelocityCommand ranges.

    Each compute, advances `cfg.ranges.{lin_vel_x,lin_vel_y,ang_vel_z}` to the
    latest stage whose `step` threshold `env.common_step_counter` has reached.
    Later stages override earlier ones.

    Also rewrites `rel_forward_envs` based on the active stage (see
    `_rel_forward_for`): non-zero while the stage is forward-only, zero as
    soon as any axis opens up (otherwise the forward-env clamp would mutilate
    multi-axis commands).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRlEnv"):
        self._stages: list[_CommandStage] = cfg.params["stages"]
        command_name: str = cfg.params.get("command_name", "twist")
        term_cfg = env.command_manager.get_term_cfg(command_name)
        assert isinstance(term_cfg, UniformVelocityCommandCfg)
        self._cmd_cfg = term_cfg
        self._validate()

    def _validate(self) -> None:
        required = {"step", "x_com", "y_com", "z_com"}
        for i, stage in enumerate(self._stages):
            missing = required - stage.keys()
            if missing:
                raise KeyError(f"Stage {i} missing required keys: {sorted(missing)}")
            if i > 0 and stage["step"] < self._stages[i - 1]["step"]:
                raise ValueError(
                    f"Curriculum stages must be nondecreasing in step; "
                    f"stage {i} has step {stage['step']} < {self._stages[i - 1]['step']}."
                )

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        stages: list[_CommandStage],
        command_name: str = "twist",
    ) -> dict[str, torch.Tensor]:
        del env_ids, stages, command_name
        step = env.common_step_counter
        active: _CommandStage | None = None
        for stage in self._stages:
            if step >= stage["step"]:
                active = stage
        if active is None:
            return {}

        self._cmd_cfg.ranges.lin_vel_x = active["x_com"]
        self._cmd_cfg.ranges.lin_vel_y = active["y_com"]
        self._cmd_cfg.ranges.ang_vel_z = active["z_com"]
        self._cmd_cfg.rel_forward_envs = _rel_forward_for(active)

        return {
            "x_lo":             torch.tensor(active["x_com"][0]),
            "x_hi":             torch.tensor(active["x_com"][1]),
            "y_lo":             torch.tensor(active["y_com"][0]),
            "y_hi":             torch.tensor(active["y_com"][1]),
            "z_lo":             torch.tensor(active["z_com"][0]),
            "z_hi":             torch.tensor(active["z_com"][1]),
            "rel_forward_envs": torch.tensor(self._cmd_cfg.rel_forward_envs),
        }


# ---------------------------------------------------------------------------
# Rough-terrain generator
# ---------------------------------------------------------------------------
def _build_rough_terrain_generator() -> TerrainGeneratorCfg:
    """Heightfield generator for blind rough-terrain training.

    `num_rows` × `num_cols` patches; difficulty scales linearly along rows
    (level 0 = flat) via mjlab's `curriculum=True` mode. mjlab's
    `terrain_levels_vel` curriculum promotes/demotes each env between rows
    based on tracking score.
    """
    t = C.TERRAIN
    return TerrainGeneratorCfg(
        size=t["size"],
        border_width=t["border_size"],
        num_rows=t["num_rows"],
        num_cols=t["num_cols"],
        curriculum=True,
        sub_terrains={
            "uniform_noise": HfRandomUniformTerrainCfg(
                proportion=1.0,
                size=t["size"],
                noise_range=t["height_range"],
                noise_step=t["height_range"][1] / 4 if t["height_range"][1] > 0 else 0.005,
                horizontal_scale=t["step_range"][0],
                vertical_scale=0.005,
                border_width=0.25,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------
def sesame_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-terrain velocity-tracking env for Sesame. Everything sourced from config.py."""
    cfg = make_velocity_env_cfg()

    # --- Scene / terrain -----------------------------------------------------
    assert cfg.scene.terrain is not None
    if C.TERRAIN["rough_enabled"] and not play:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = _build_rough_terrain_generator()
        cfg.scene.terrain.max_init_terrain_level = 0
    else:
        # Flat. Plane mode also disables the rough-terrain curriculum below
        # since `terrain_levels_vel` requires `terrain_levels` state on the
        # terrain entity, which only generator-mode populates.
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    cfg.scene.entities = {"robot": get_robot_cfg()}
    cfg.scene.num_envs = 1 if play else C.NUM_ENVS

    # --- Sensors -------------------------------------------------------------
    # Drop rough-terrain raycast sensors the base config carries.
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in {"terrain_scan", "foot_height_scan"}
    )
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = cfg.scene.sensors + (feet_ground,)

    # --- Observations --------------------------------------------------------
    # Drop terms tied to sensors we no longer have.
    for group in ("actor", "critic"):
        for key in ("height_scan", "foot_height", "foot_air_time",
                    "foot_contact", "foot_contact_forces"):
            cfg.observations[group].terms.pop(key, None)

    # Apply OBSERVATIONS toggles to both groups.
    for key, spec in C.OBSERVATIONS.items():
        if not spec.get("enabled", True):
            cfg.observations["actor"].terms.pop(key, None)
            cfg.observations["critic"].terms.pop(key, None)

    # --- Actions -------------------------------------------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ACTION_SCALE

    # --- Commands ------------------------------------------------------------
    # Seed ranges from curriculum stage 0; the curriculum term (registered
    # below) owns them from then on.
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    stage0 = C.COMMAND_CURRICULUM[0]
    final_stage = C.COMMAND_CURRICULUM[-1]
    twist.ranges.lin_vel_x = stage0["x_com"]
    twist.ranges.lin_vel_y = stage0["y_com"]
    twist.ranges.ang_vel_z = stage0["z_com"]
    twist.ranges.heading = None
    twist.heading_command = False
    twist.rel_standing_envs = C.REL_STANDING_ENVS
    twist.rel_forward_envs = _rel_forward_for(stage0)

    # --- Rewards -------------------------------------------------------------
    # Drop base rewards that depend on terrain sensors we removed.
    for key in ("pose", "foot_clearance", "foot_swing_height"):
        cfg.rewards.pop(key, None)

    # Wire per-robot body/site names on inherited rewards.
    cfg.rewards["upright"].params["asset_cfg"].body_names = (C.BASE_LINK_NAME,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (C.BASE_LINK_NAME,)
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

    # Tighten linear-velocity tracking kernel so small commands stop giving
    # near-full reward for standing still. mjlab default is sqrt(0.25) = 0.5.
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.05)

    # Register electrical-power penalty (not in mjlab's base velocity cfg).
    cfg.rewards["electrical_power"] = RewardTermCfg(
        func=envs_mdp.electrical_power_cost,
        weight=0.0,  # set below from config
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )

    # Default-pose attractor: keeps all joints close to DEFAULT_JOINT_ANGLES.
    cfg.rewards["posture"] = RewardTermCfg(
        func=envs_mdp.posture,
        weight=0.0,  # set below from config
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "std": {".*": 0.3},
        },
    )

    # 2nd-order action penalty (jerk); complements action_rate_l2.
    cfg.rewards["action_acc_l2"] = RewardTermCfg(
        func=envs_mdp.action_acc_l2,
        weight=0.0,  # set below from config
    )

    # Apply REWARDS toggles / weights.
    for cfg_name, spec in C.REWARDS.items():
        mjlab_name = _MJLAB_REWARD_NAME[cfg_name]
        if not spec.get("enabled", True):
            cfg.rewards.pop(mjlab_name, None)
            continue
        if mjlab_name in cfg.rewards:
            cfg.rewards[mjlab_name].weight = spec["weight"]

    # # Zero out body_ang_vel / angular_momentum so they stick around for CLI
    # # overrides but contribute nothing by default.
    # for name in ("body_ang_vel", "angular_momentum"):
    #     if name in cfg.rewards:
    #         cfg.rewards[name].weight = 0.0

    # air_time gate must be < commanded |v| or the reward never fires.
    if "air_time" in cfg.rewards:
        cmd_threshold = C.REWARDS["air_time"].get("cmd_threshold", 0.1)
        cfg.rewards["air_time"].params["command_threshold"] = cmd_threshold

    # --- Events --------------------------------------------------------------
    for name, spec in C.EVENTS.items():
        if not spec.get("enabled", True) or (name == "push_robot" and play):
            cfg.events.pop(name, None)

    if C.EVENTS["foot_friction"]["enabled"]:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
            "robot", geom_names=FOOT_GEOM_NAMES
        )
    if C.EVENTS["base_com"]["enabled"]:
        cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=(C.BASE_LINK_NAME,)
        )

    # --- Terminations --------------------------------------------------------
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("fell_over", None)
    if C.TERMINATE_ON_FALL:
        cfg.terminations["fell_over"] = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(C.FALL_ANGLE_DEG)},
        )

    # --- Curriculum ----------------------------------------------------------
    cfg.curriculum = {}
    if C.CURRICULUM_ENABLED and not play:
        cfg.curriculum["command_ranges"] = CurriculumTermCfg(
            func=command_range_curriculum,
            params={
                "stages": list(C.COMMAND_CURRICULUM),
                "command_name": "twist",
            },
        )
    if C.TERRAIN["rough_enabled"] and not play:
        # Per-env terrain-level promotion: envs that successfully track their
        # commanded velocity over an episode are bumped to a harder difficulty
        # band; envs that fall short get bumped down. Caps at num_rows-1.
        from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_vel,
            params={"command_name": "twist"},
        )

    # --- Viewer / sim --------------------------------------------------------
    cfg.viewer.body_name = C.BASE_LINK_NAME
    cfg.viewer.distance = 0.8
    cfg.viewer.elevation = -15.0

    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    cfg.episode_length_s = C.EPISODE_LENGTH_S

    # --- Play-mode tweaks ----------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.scene.extent = 1.0

        # Use the full final-stage ranges so the viser joystick lets the user
        # drag across both directions (important for symmetric policies). A
        # single initial sample is drawn and then pinned via resampling_time.
        # Each slider max is nudged to >= 0.1 to satisfy viser's assertion.
        def _nudge(lo: float, hi: float) -> tuple[float, float]:
            return (lo, max(hi, 0.1))

        twist.ranges.lin_vel_x = _nudge(*final_stage["x_com"])
        twist.ranges.lin_vel_y = _nudge(*final_stage["y_com"])
        twist.ranges.ang_vel_z = _nudge(*final_stage["z_com"])
        twist.rel_forward_envs = 0.0   # never abs-clamp in play
        twist.rel_standing_envs = 0.0
        twist.resampling_time_range = (1e6, 1e6)

    return cfg


def build_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Translate config.PPO into mjlab's typed runner cfg."""
    p = C.PPO
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(**p["actor"]),
        critic=RslRlModelCfg(**p["critic"]),
        algorithm=RslRlPpoAlgorithmCfg(**p["algorithm"]),
        experiment_name=C.EXPERIMENT_NAME,
        save_interval=p["save_interval"],
        num_steps_per_env=p["num_steps_per_env"],
        max_iterations=C.MAX_ITERATIONS,
        logger=p["logger"],
    )
