"""Train PPO on the Sesame forward-velocity task.

Usage:
  python train.py                                                  # full run
  python train.py --max-iterations 20                              # smoke test
  python train.py --num-envs 2048                                  # override NUM_ENVS
  python train.py --resume                                         # latest run under logs/
  python train.py --checkpoint logs/<run>/model_<N>.pt             # pin a specific checkpoint
  python train.py --checkpoint <path> --continue-iteration         # keep the prior iteration counter
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from mjlab.scripts.train import TrainConfig, run_train
from mjlab.utils.gpu import select_gpus

import config as C
from env_cfg import register_all_tasks

# Registers both Sesame-Velocity-Flat and Sesame-Velocity-Rough. Training
# picks one via C.TASK_ID (resolved from TERRAIN["rough_enabled"]).
register_all_tasks()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Sesame-Velocity-Flat.")
    p.add_argument("--max-iterations", type=int, default=C.MAX_ITERATIONS,
                   help="PPO training iterations (default from config.MAX_ITERATIONS).")
    p.add_argument("--num-envs", type=int, default=C.NUM_ENVS,
                   help="Parallel environments (default from config.NUM_ENVS).")
    p.add_argument("--resume", action="store_true",
                   help="Continue PPO from the latest checkpoint under logs/.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Direct path to a .pt file (relative or absolute). "
                        "Implies --resume.")
    p.add_argument("--continue-iteration", action="store_true",
                   help="Load weights+optimizer from the checkpoint and use the last step counter.")
    return p.parse_args()


def _apply_reset_iteration_patch() -> None:
    """Make the runner ignore the saved iteration counter on load.
    Only resets the iteration counter for tensorboard.
    Step counter for curriculum remains the same.
    """
    from mjlab.rl.runner import MjlabOnPolicyRunner

    _orig_load = MjlabOnPolicyRunner.load

    def _load_no_iter(self, path, *args, **kwargs):
        result = _orig_load(self, path, *args, **kwargs)
        self.current_learning_iteration = 0
        print("[train] zeroed current_learning_iteration after load")
        return result

    MjlabOnPolicyRunner.load = _load_no_iter


def main() -> None:
    args = _parse_args()

    cfg = TrainConfig.from_task(C.TASK_ID)
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.seed = C.SEED
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.seed = C.SEED

    if args.checkpoint is not None:
        ckpt = args.checkpoint.resolve()
        if not ckpt.exists():
            raise SystemExit(f"--checkpoint not found: {ckpt}")
        cfg.agent.resume = True
        cfg.agent.load_run = ckpt.parent.name
        cfg.agent.load_checkpoint = ckpt.name
        print(f"[train] resuming from {ckpt}")
    else:
        cfg.agent.resume = args.resume

    if cfg.agent.resume and not args.continue_iteration:
        _apply_reset_iteration_patch()
    elif args.continue_iteration and not cfg.agent.resume:
        print("[train] --continue-iteration ignored: no checkpoint to resume from.")

    log_dir = Path("logs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Single-GPU launch.
    selected_gpus, num_gpus = select_gpus([0])
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    )
    os.environ["MUJOCO_GL"] = "egl"
    run_train(C.TASK_ID, cfg, log_dir)


if __name__ == "__main__":
    main()
