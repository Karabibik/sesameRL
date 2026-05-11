"""Play (visualize) the Sesame forward-velocity task.

Usage:
  python play.py                                 # auto-loads latest checkpoint
  python play.py --checkpoint-file path/to.pt    # pin a specific checkpoint
  python play.py --viewer native                 # native mujoco viewer
  python play.py --terrain rough                 # Perlin terrain instead of plane
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mjlab.scripts.play import PlayConfig, run_play
from mjlab.utils.os import get_checkpoint_path

import config as C
from env_cfg import register_all_tasks

# Registers both Sesame-Velocity-Flat and Sesame-Velocity-Rough.
register_all_tasks()


def _latest_checkpoint() -> str | None:
    """Newest `model_*.pt` under logs/<timestamp>/; None if none exists.

    Match only datetime-shaped run dirs so legacy folders (e.g. a stray
    `logs/rsl_rl/` from an older layout) don't hijack resolution.
    """
    logs_root = Path("logs")
    if not logs_root.exists():
        return None
    run_dir_pat = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.*"
    try:
        return str(get_checkpoint_path(logs_root, run_dir_pat, r"model_\d+\.pt"))
    except ValueError:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play a trained Sesame policy.")
    p.add_argument("--checkpoint-file", type=str, default=None,
                   help="Path to a model_*.pt. Defaults to the newest under logs/.")
    p.add_argument("--viewer", choices=("viser", "native"), default="viser",
                   help="viser (web) or native mujoco viewer.")
    p.add_argument("--terrain", choices=("flat", "rough"), default="flat",
                   help="Which Sesame variant to play. Defaults to flat.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint = args.checkpoint_file or _latest_checkpoint()
    if checkpoint and not args.checkpoint_file:
        print(f"[play] auto-loading latest checkpoint: {checkpoint}")

    task_id = {
        "flat":  C.TASK_FLAT,
        "rough": C.TASK_ROUGH,
        None:    C.TASK_ID,
    }[args.terrain]
    print(f"[play] task_id: {task_id}")

    cfg = PlayConfig(
        agent="trained" if checkpoint else "zero",
        num_envs=1,
        viewer=args.viewer,
        checkpoint_file=checkpoint,
    )
    run_play(task_id, cfg)


if __name__ == "__main__":
    main()
