"""Sesame robot spec: URDF loading, actuators, init state, collisions.

Reads everything from [config.py](config.py). Produces the mjlab `EntityCfg`
consumed by `sesame_flat_env_cfg` in [env_cfg.py](env_cfg.py).

Run this file directly to launch a passive MuJoCo viewer on the compiled
spec -- a quick check that the URDF, meshes, sensors and sites all load
cleanly before spending any compute on RL.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

import config as C


URDF_PATH: Path = Path(__file__).resolve().parent / C.URDF_PATH
assert URDF_PATH.exists(), URDF_PATH


# IMU mount point (torso CoM) and per-foot site positions in their distal-
# link frames. Sign on z flips between L/R sides because the mirrored legs
# live in different halves of the joint range.
_IMU_POS = (-0.02315, -0.020775, -0.024241)
_FOOT_SITES: dict[str, tuple[str, tuple[float, float, float]]] = {
    "Link_L3": ("foot_FL", (0.0046158, -0.00016469, -0.035)),
    "Link_L4": ("foot_RL", (0.0059082, -0.000077912, 0.035)),
    "Link_R3": ("foot_FR", (0.00607, -0.00014231, 0.035)),
    "Link_R4": ("foot_RR", (0.0045634, -0.000071454, -0.035)),
}
_FOOT_BODIES = set(_FOOT_SITES.keys())

FOOT_GEOM_NAMES = tuple(f"{b}_foot_collision" for b in _FOOT_BODIES)
FOOT_SITE_NAMES = tuple(s for _, (s, _) in _FOOT_SITES.items())


def get_spec() -> mujoco.MjSpec:
    """Build an MjSpec from the URDF and patch in mjlab-required extras."""
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))

    # URDF uses package://meshes/...
    for mesh in spec.meshes:
        mesh.file = mesh.file.rsplit("/", 1)[-1]
    spec.compiler.meshdir = str(URDF_PATH.parent / "meshes")
    spec.compiler.autolimits = True

    # Name each collision geom so CollisionCfg regexes can target feet only.
    for body in spec.bodies:
        for geom in body.geoms:
            if body.name in _FOOT_BODIES:
                geom.name = f"{body.name}_foot_collision"
            elif body.name.startswith("Link_") or body.name == C.BASE_LINK_NAME:
                geom.name = f"{body.name}_collision"

    # Free-floating base (URDF is fixed-base by default).
    base = spec.body(C.BASE_LINK_NAME)
    base.add_freejoint(name="floating_base")

    # Reference sites: IMU on torso, foot tips on each lower leg.
    base.add_site(name="imu", pos=list(_IMU_POS), group=5)
    for body_name, (site_name, pos) in _FOOT_SITES.items():
        spec.body(body_name).add_site(
            name=site_name, pos=list(pos), size=[0.003, 0.0, 0.0], group=5
        )

    # Onboard sensors consumed by observations / rewards.
    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_GYRO, name="imu_ang_vel",
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
    )
    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER, name="imu_lin_vel",
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
    )
    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, name="imu_lin_acc",
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
    )
    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, name="root_angmom",
        objtype=mujoco.mjtObj.mjOBJ_BODY, objname=C.BASE_LINK_NAME,
    )
    return spec


# Two actuator groups, one per joint pattern. Sharing stiffness/damping keeps
# the tuning story simple; split them here if you want per-group gains.
HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(C.HIP_JOINT_REGEX,),
    stiffness=C.STIFFNESS,
    damping=C.DAMPING,
    effort_limit=C.EFFORT_LIMIT,
    armature=C.ARMATURE,
)
KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(C.KNEE_JOINT_REGEX,),
    stiffness=C.STIFFNESS,
    damping=C.DAMPING,
    effort_limit=C.EFFORT_LIMIT,
    armature=C.ARMATURE,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=C.BASE_INIT_POS,
    joint_pos=dict(C.DEFAULT_JOINT_ANGLES),
    joint_vel={".*": 0.0},
)

# Only the feet touch the ground; other collision geoms are silenced by
# CollisionCfg so we don't pay for self-collision or torso-vs-ground.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=("Link_[LR][34]_foot_collision",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.8, 0.01, 0.001),
    solimp=(0.9, 0.95, 0.023),
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(HIP_ACTUATOR_CFG, KNEE_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9,
)


def get_robot_cfg() -> EntityCfg:
    """Fresh EntityCfg for Sesame. New instance each call to avoid shared mutation."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# dict of actuator-regex -> per-actuator action scale. Used by the joint_pos
# action term to map raw policy output +-1 into a radian delta from default.
ACTION_SCALE: dict[str, float] = {
    C.HIP_JOINT_REGEX: C.ACTION_SCALE,
    C.KNEE_JOINT_REGEX: C.ACTION_SCALE,
}


if __name__ == "__main__":
    import mujoco.viewer
    mujoco.viewer.launch(get_spec().compile())
