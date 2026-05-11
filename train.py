"""Train PPO on the Sesame forward-velocity task.

Usage:
  python train.py                              # full run, config defaults
  python train.py --max-iterations 20          # smoke test
  python train.py --run-name my_experiment     # tag the log subdir
  python train.py --num-envs 2048              # override NUM_ENVS from config
  python train.py --resume                     # continue from latest run under logs/
  python train.py --resume --load-run 2026-05-09_14-30-00
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.gpu import select_gpus

import config as C
from env_cfg import build_rl_cfg, sesame_flat_env_cfg


register_mjlab_task(
    task_id=C.TASK_ID,
    env_cfg=sesame_flat_env_cfg(),
    play_env_cfg=sesame_flat_env_cfg(play=True),
    rl_cfg=build_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Sesame-Velocity-Flat.")
    p.add_argument("--max-iterations", type=int, default=C.MAX_ITERATIONS,
                   help="PPO training iterations (default from config.MAX_ITERATIONS).")
    p.add_argument("--run-name", type=str, default="",
                   help="Suffix appended to the logs/<timestamp>[_<run-name>]/ dir.")
    p.add_argument("--num-envs", type=int, default=C.NUM_ENVS,
                   help="Parallel environments (default from config.NUM_ENVS).")
    p.add_argument("--seed", type=int, default=C.SEED,
                   help="Random seed for env + agent.")
    p.add_argument("--resume", action="store_true",
                   help="Continue PPO from a prior checkpoint under logs/.")
    p.add_argument("--load-run", type=str, default="",
                   help="Run dir under logs/<EXPERIMENT_NAME>/ to resume from "
                        "(default: latest).")
    p.add_argument("--load-checkpoint", type=str, default="",
                   help="Checkpoint filename to load (default: latest model_*.pt).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = TrainConfig.from_task(C.TASK_ID)
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.run_name = args.run_name
    cfg.agent.seed = args.seed
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.seed = args.seed

    cfg.agent.resume = args.resume
    cfg.agent.load_run = args.load_run
    cfg.agent.load_checkpoint = args.load_checkpoint

    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        log_dir_name += f"_{args.run_name}"
    log_dir = Path("logs") / log_dir_name

    # Single-GPU launch.
    selected_gpus, num_gpus = select_gpus([0])
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    )
    os.environ["MUJOCO_GL"] = "egl"
    run_train(C.TASK_ID, cfg, log_dir)


if __name__ == "__main__":
    main()
