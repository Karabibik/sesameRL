"""Central configuration for sesameRL (mjlab + rsl_rl).

Everything experiment-relevant lives here. Toggle a term on or off with its
`enabled` flag; tune weights, params or curriculum stages in place.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Robot -- Sesame has 8 revolute joints (hip + knee per leg). Leg mapping:
#   L3 = Front-Left    L4 = Rear-Left
#   R3 = Front-Right   R4 = Rear-Right
# ---------------------------------------------------------------------------
URDF_PATH = "urdf_model/Sesame.urdf"
BASE_LINK_NAME = "base_link"

HIP_JOINT_REGEX = "Joint_[LR][12]"
KNEE_JOINT_REGEX = "Joint_[LR][34]"

# The URDF ships asymmetric joint limits: four joints use [0, pi/2], the other four 
# use [pi/2, pi]. The neutral stand pose is the midpoint of each hip range.
_HIP_LOW_MID = math.pi / 4              # joints with range [0, pi/2]
_HIP_HIGH_MID = 3 * math.pi / 4         # joints with range [pi/2, pi]
_KNEE_LOW_EXT = math.pi / 4             # knees  with range [0, pi/2]
_KNEE_HIGH_EXT = 3 * math.pi / 4        # knees  with range [pi/2, pi]

DEFAULT_JOINT_ANGLES = {
    "Joint_L1": _HIP_LOW_MID,
    "Joint_L2": _HIP_HIGH_MID,
    "Joint_R1": _HIP_HIGH_MID,
    "Joint_R2": _HIP_LOW_MID,
    "Joint_L3": _KNEE_LOW_EXT,
    "Joint_L4": _KNEE_HIGH_EXT,
    "Joint_R3": _KNEE_HIGH_EXT,
    "Joint_R4": _KNEE_LOW_EXT,
}

BASE_INIT_POS = (0.0, 0.0, 0.06)

# Actuator gains (PD servo per joint). effort/velocity limits come from the URDF
EFFORT_LIMIT = 0.215      # Nm
VELOCITY_LIMIT = 10.47    # rad/s

# stiffness/damping tuned so action_scale ~= 0.4 rad.
# ![TODO] Tuned somewhat manually, open to edit
STIFFNESS = 0.14
DAMPING = 0.003
ARMATURE = 1e-5

# Per-actuator action scale. q_target = default_pose + scale * action.
# ?[OPT] Didn't want to fully extend the legs
ACTION_SCALE = 0.6 #0.25 * EFFORT_LIMIT / STIFFNESS


# ---------------------------------------------------------------------------
# Scene / simulation
# ---------------------------------------------------------------------------
NUM_ENVS = 4096 # ?[OPT] Check your VRAM, 8192 envs almost fill 8GB
EPISODE_LENGTH_S = 20
FALL_ANGLE_DEG = 60.0          # episode termination tilt threshold
TERMINATE_ON_FALL = True


# ---------------------------------------------------------------------------
# Terrain. `rough_enabled=False` keeps the existing flat-plane scene unchanged.
# Recommended: train flat first, then `python train.py --resume` with this on.
# ---------------------------------------------------------------------------
TERRAIN = {
    "rough_enabled":  False,
    "height_range":   (0.0, 0.2),    # Perlin amplitude in m.
    "size":           (8.0, 8.0),    # patch size, m.
    "border_size":    5.0,           # flat margin around each patch, m.
    "num_rows":       5,             # difficulty bands.
    "num_cols":       5,             # variants per level.
}


# ---------------------------------------------------------------------------
# Observations. The `actor` and `critic` observation groups both see the same set; 
# only the actor group gets the configured observation-noise corruption.
# ---------------------------------------------------------------------------
# ?[OPT] No need for observation history, this seems fine but open to edits
OBSERVATIONS = {
    
    # Needs IMU
    "base_lin_vel":      {"enabled": True},
    "base_ang_vel":      {"enabled": True},
    "projected_gravity": {"enabled": True},

    # Needs encoder
    "joint_pos":         {"enabled": True},  # delta from default pose
    "joint_vel":         {"enabled": True},
    
    "actions":           {"enabled": True},  # previous action
    "command":           {"enabled": True},  # (vx, vy, wz) target
}


# ---------------------------------------------------------------------------
# Rewards -- weights are passed straight to mjlab's RewardManager.
# ---------------------------------------------------------------------------
REWARDS = {
    "track_linear_velocity":  {"enabled": True, "weight":  2.0},
    "track_angular_velocity": {"enabled": True, "weight":  1.5}, # Making same with linear causes motion leak
    "upright":                {"enabled": True, "weight":  1.0},
    "action_rate_l2":         {"enabled": True, "weight": -0.1},
    "action_acc_l2":          {"enabled": True, "weight": -0.005},
    "foot_slip":              {"enabled": True, "weight": -0.05},
    "dof_pos_limits":         {"enabled": True, "weight": -1.0},
    "soft_landing":           {"enabled": True, "weight": -1e-5},
    "air_time":               {"enabled": True, "weight":  0.5, "cmd_threshold": 0.05},

    # exp(-mean((q - q_default)^2 / std^2));
    # default-pose attractor, keeps limbs from flailing asymmetrically.
    "posture":                {"enabled": True, "weight":  0.3},

    # sum(max(tau*qdot, 0)) over all 8 joints; discourages thrashing.
    "electrical_power":       {"enabled": True, "weight": -2e-4},
}


# ---------------------------------------------------------------------------
# Events (domain randomization & perturbations)
# ---------------------------------------------------------------------------
# ![TODO] Play with these
EVENTS = {
    "push_robot":    {"enabled": False},  # enable once un-perturbed policy walks
    "encoder_bias":  {"enabled": True},
    "foot_friction": {"enabled": True},
    "base_com":      {"enabled": True},
}


# ---------------------------------------------------------------------------
# Commands -- UniformVelocityCommand over (lin_vel_x, lin_vel_y, ang_vel_z).
# iteration*24, where iteration is the number of steps seen in tensorboard
# 24 = num_steps_per_env in PPO config
# ---------------------------------------------------------------------------
CURRICULUM_ENABLED = True
COMMAND_CURRICULUM = [
    # Stage 0: symmetric forward/backward only, narrow range. Easy warmup.
    {"step":        0, "x_com": (-0.3, 0.3), "y_com": ( 0.0, 0.0), "z_com": ( 0.0, 0.0)},
    # Stage 1: widen forward/backward.
    {"step":   500*24, "x_com": (-0.5, 0.5), "y_com": ( 0.0, 0.0), "z_com": ( 0.0, 0.0)},
    # Stage 2: add lateral strafe.
    {"step":  1000*24, "x_com": (-0.5, 0.5), "y_com": (-0.3, 0.3), "z_com": ( 0.0, 0.0)},
    # Stage 3: full symmetric 3-axis.
    {"step":  1500*24, "x_com": (-0.75, 0.75), "y_com": (-0.75, 0.75), "z_com": (-2.0, 2.0)},

]
REL_STANDING_ENVS = 0.1


# ---------------------------------------------------------------------------
# Training (PPO via rsl_rl)
# ---------------------------------------------------------------------------
TASK_FLAT  = "Sesame-Velocity-Flat"
TASK_ROUGH = "Sesame-Velocity-Rough"
TASK_ID    = TASK_ROUGH if TERRAIN["rough_enabled"] else TASK_FLAT
EXPERIMENT_NAME = "sesame_velocity"
MAX_ITERATIONS = 20_000
SEED = 1

PPO = {
    "num_steps_per_env": 24,
    "save_interval": 200,
    "logger": "tensorboard",

    "algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },

    "actor": {
        "hidden_dims": (256, 128),
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    },
    "critic": {
        "hidden_dims": (256, 128),
        "activation": "elu",
        "obs_normalization": False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def enabled_rewards() -> dict:
    return {k: v for k, v in REWARDS.items() if v.get("enabled", True)}


def enabled_observations() -> list[str]:
    return [k for k, v in OBSERVATIONS.items() if v.get("enabled", True)]


def enabled_events() -> list[str]:
    return [k for k, v in EVENTS.items() if v.get("enabled", True)]
