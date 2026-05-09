"""Copy the latest auto-exported ONNX policy from logs/ into web/assets/.

mjlab's `VelocityOnPolicyRunner.save()` writes a `<timestamp>.onnx` next to
each `model_*.pt` checkpoint. This script picks the one in the newest run
directory (matching `play.py`'s logic) and copies it to a stable path the
demo loads from.

Usage:
  python web/sync_policy.py                       # latest run, latest ONNX
  python web/sync_policy.py --onnx <path.onnx>    # pin a specific ONNX
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
DEST = ROOT / "web" / "assets" / "policy.onnx"

_RUN_DIR_PAT = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.*")


def _latest_onnx() -> Path:
    runs = sorted(
        (p for p in LOGS.iterdir() if p.is_dir() and _RUN_DIR_PAT.fullmatch(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise SystemExit(f"no datetime-shaped run dirs under {LOGS}")
    for run in runs:
        candidates = sorted(run.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    raise SystemExit(f"no .onnx files under any run dir in {LOGS}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, default=None, help="Specific ONNX file to sync.")
    args = p.parse_args()

    src = args.onnx or _latest_onnx()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, DEST)
    print(f"[sync_policy] {src} -> {DEST}")


if __name__ == "__main__":
    main()
