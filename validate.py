"""Sinusoidal joint validation rollout -- no RL, no policy.

Drives every joint with a sinusoid around its default pose to confirm:
  1. the URDF + meshes load,
  2. the freejoint behaves,
  3. actuators track their targets,
  4. contacts are stable.

Run this BEFORE spending compute on PPO.

Usage:
  python validate.py
  python validate.py --viewer native
  python validate.py --freq-hz 1.5 --amplitude 0.8 --no-trot
"""

from __future__ import annotations

import argparse
import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import config as C
from env_cfg import sesame_flat_env_cfg


def _build_phases(joint_names: list[str], trot: bool) -> torch.Tensor:
    """Trot gait: FL+RR in phase, FR+RL anti-phase.

    Leg naming: R3=front-right, R4=rear-right, L3=front-left, L4=rear-left.
    Hip joints drive their child: L1->L3 (FL hip), L2->L4 (RL hip),
    R1->R3 (FR hip), R2->R4 (RR hip).
    FL+RR in phase  <=> (L1, L3, R2, R4)
    FR+RL opposite <=> (R1, R3, L2, L4)
    """
    if not trot:
        return torch.zeros(len(joint_names))
    pair_a = {"Joint_L1", "Joint_L3", "Joint_R2", "Joint_R4"}
    phases = [0.0 if n in pair_a else math.pi for n in joint_names]
    return torch.tensor(phases, dtype=torch.float32)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinusoidal joint validator (no policy).")
    p.add_argument("--freq-hz", type=float, default=1.5)
    p.add_argument("--amplitude", type=float, default=0.8,
                   help="Raw action units; final swing = amplitude * action_scale.")
    p.add_argument("--no-trot", dest="trot", action="store_false",
                   help="Disable trot phasing (all joints in phase).")
    p.add_argument("--viewer", choices=("viser", "native"), default="viser")
    p.set_defaults(trot=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Disable fall-termination so the robot can just keep swinging even if it
    # tips over; validate.py is about physics, not policy.
    orig_terminate = C.TERMINATE_ON_FALL
    orig_num_envs = C.NUM_ENVS
    C.TERMINATE_ON_FALL = False
    C.NUM_ENVS = 1

    try:
        env_cfg = sesame_flat_env_cfg(play=True)
        device = "cuda"
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

        joint_names = env.action_manager.get_term("joint_pos").target_names
        phases = _build_phases(joint_names, args.trot).to(device)
        print(f"[validate] joint order: {joint_names}")

        dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
        step_counter = {"n": 0}

        def sin_policy(_obs) -> torch.Tensor:
            t = step_counter["n"] * dt
            step_counter["n"] += 1
            action = args.amplitude * torch.sin(
                2 * math.pi * args.freq_hz * t + phases
            )
            return action.unsqueeze(0).to(device)

        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        viewer_cls = ViserPlayViewer if args.viewer == "viser" else NativeMujocoViewer
        viewer_cls(wrapped, sin_policy).run()
        wrapped.close()
    finally:
        C.TERMINATE_ON_FALL = orig_terminate
        C.NUM_ENVS = orig_num_envs


if __name__ == "__main__":
    main()
