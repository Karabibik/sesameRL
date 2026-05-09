"""mjswan demo entrypoint for the Sesame quadruped policy.

Local:
    python web/main.py                # builds web/dist and launches at localhost:8080
    python web/main.py --no-launch    # build only

CI / GitHub Pages:
    MJSWAN_BASE_PATH="/<repo>/" MJSWAN_NO_LAUNCH=1 python web/main.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import onnx

import mjswan

# mjswan hardcodes the cursor-drag force at 100 N inside its TS runtime, which
# is way too strong for the ~1 kg Sesame. We patch its template idempotently
# to a robot-appropriate value before building. The patch survives until
# `pip install --upgrade mjswan` overwrites the template; re-running this
# script re-applies it.
DRAG_FORCE_SCALE = 5.0

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Side effect: registers `Sesame-Velocity-Flat` in the mjlab task registry.
import play  # noqa: E402,F401

import config as C  # noqa: E402
from env_cfg import sesame_flat_env_cfg  # noqa: E402

ASSETS = ROOT / "web" / "assets"
DIST = ROOT / "web" / "dist"


def _robust_rmtree(path: Path, retries: int = 5, delay: float = 0.5) -> None:
    """rmtree that survives Windows NTFS races.

    `node_modules` from mjswan's vite build contains thousands of tiny files
    (e.g. @tabler/icons-react), and shutil.rmtree on Windows occasionally
    raises ``WinError 145: directory is not empty`` because a child rmdir
    completes a hair after the parent rmdir is attempted. Retry a handful
    of times with backoff before giving up.
    """
    if not path.exists():
        return
    import stat
    import time

    last_failure: tuple[Path, BaseException] | None = None

    def _onerror(func, p, excinfo):
        nonlocal last_failure
        # First, try the read-only-attribute fix (common for .git internals).
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
            return
        except OSError as e:
            last_failure = (Path(p), e)

    for attempt in range(retries):
        last_failure = None
        try:
            shutil.rmtree(path, onerror=_onerror)
        except OSError as e:
            last_failure = (path, e)
        if not path.exists():
            return
        time.sleep(delay * (attempt + 1))

    # Walk the leftover tree to give the user a useful pointer.
    leftover: list[str] = []
    for root, _dirs, files in os.walk(path):
        for f in files:
            leftover.append(os.path.join(root, f))
            if len(leftover) >= 3:
                break
        if len(leftover) >= 3:
            break
    msg = f"Could not remove {path} after {retries} attempts."
    if last_failure is not None:
        msg += f"\n  Last failure: {last_failure[1]} on {last_failure[0]}"
    if leftover:
        msg += "\n  Files still present (likely locked by another process):"
        for f in leftover:
            msg += f"\n    {f}"
        msg += "\n  Close any running `python web/main.py` dev server before retrying."
    raise RuntimeError(msg)


def _patch_drag_force_scale(scale: float) -> None:
    """Idempotently rewrite the hardcoded `dragForceScale` in mjswan's template.

    mjswan exposes no Python or runtime hook for cursor-drag strength
    (it's a private field initialized to 100.0 in
    `template/src/core/engine/runtime.ts`). For a small robot like
    Sesame we want it much lower. Edit the source in place; the next
    vite build will pick up the change. This mutates the venv's
    mjswan install -- acceptable for a per-project demo, and a no-op
    if the file already has the desired value.
    """
    rt = (
        Path(mjswan.__file__).parent
        / "template" / "src" / "core" / "engine" / "runtime.ts"
    )
    if not rt.exists():
        print(f"[main] WARN: cannot find {rt}; skipping drag-force patch.")
        return

    text = rt.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"this\.dragForceScale\s*=\s*[\d.]+\s*;",
        f"this.dragForceScale = {scale};",
        text,
        count=1,
    )
    if n == 0:
        print("[main] WARN: dragForceScale assignment not found; mjswan upstream may have changed.")
        return
    if new_text == text:
        return  # already patched
    rt.write_text(new_text, encoding="utf-8")
    print(f"[main] patched mjswan dragForceScale -> {scale}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-launch", action="store_true",
                   help="Build dist/ only, skip the dev server. "
                        "Equivalent to MJSWAN_NO_LAUNCH=1.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", type=str, default="localhost")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    no_launch = args.no_launch or bool(os.environ.get("MJSWAN_NO_LAUNCH"))
    base_path = os.environ.get("MJSWAN_BASE_PATH", "/")

    _patch_drag_force_scale(DRAG_FORCE_SCALE)

    onnx_path = ASSETS / "policy.onnx"
    if not onnx_path.exists():
        raise SystemExit(
            f"missing {onnx_path} -- run `python web/sync_policy.py` first."
        )

    policy = onnx.load(str(onnx_path))
    md = {p.key: p.value for p in policy.metadata_props}

    # Pull obs/action configs straight from the registered task. mjswan's
    # mjlab adapter translates these to its own types automatically.
    play_env_cfg = sesame_flat_env_cfg(play=True)

    # The policy's action_scale is baked at training time and recorded in the
    # ONNX metadata. The live config may have drifted since (e.g. tuning for a
    # newer training run) -- override the env_cfg to match the deployed policy.
    if "action_scale" in md:
        scales = [float(x) for x in md["action_scale"].split(",")]
        if scales and all(s == scales[0] for s in scales):
            play_env_cfg.actions["joint_pos"].scale = scales[0]
            print(f"[main] action_scale overridden to {scales[0]} (from ONNX metadata)")

    # mjswan's PolicyRunner *requires* `policy_joint_names` and uses
    # `default_joint_pos` when the action term has `use_default_offset=True`.
    # Both are written into ONNX metadata by mjlab's exporter; pull them out
    # so the browser runtime can map action[i] -> the right actuator and
    # apply the correct neutral pose.
    bare_joint_names = md["joint_names"].split(",")
    # mjlab namespaces actuators/joints with the entity name (`robot/`); the
    # model lookup in PolicyStateBuilder is by exact match, so prefix to match.
    policy_joint_names = [f"robot/{n}" for n in bare_joint_names]
    default_joint_pos = [float(x) for x in md["default_joint_pos"].split(",")]

    # mjlab's `UniformVelocityCommandCfg` has no auto-mapping in mjswan, so we
    # supply mjswan's built-in `velocity_command` UI — three sliders driving
    # (lin_vel_x, lin_vel_y, ang_vel_z) — with ranges pulled from the live env.
    twist_ranges = play_env_cfg.commands["twist"].ranges
    commands = {
        "twist": mjswan.velocity_command(
            lin_vel_x=tuple(twist_ranges.lin_vel_x),
            lin_vel_y=tuple(twist_ranges.lin_vel_y),
            ang_vel_z=tuple(twist_ranges.ang_vel_z),
            default_lin_vel_x=0.0,
        ),
    }

    builder = mjswan.Builder(base_path=base_path)
    project = builder.add_project(name="sesameRL")
    scene = project.add_mjlab_scene(C.TASK_ID, play=True)

    # mjswan's PolicyRunner picks which observation group to feed the ONNX
    # session by matching keys against `OnnxModule.inKeys`, which is sourced
    # from the policy-config JSON's `onnx.meta.in_keys` and **defaults to
    # ['policy']** when absent (see template/src/core/policy/OnnxModule.ts).
    # The actual ONNX tensor input name is not consulted. mjlab uses
    # 'actor'/'critic' for its obs groups; we rekey to 'policy' so the runtime
    # picks up our single obs group. (The 'critic' group is unused at
    # inference; drop it.) Confirmed against examples/mjlab/unitree_rl/main.py.
    observations = {"policy": play_env_cfg.observations["actor"]}

    scene.add_policy(
        name="velocity",
        policy=policy,
        observations=observations,
        commands=commands,
        actions=play_env_cfg.actions,
        policy_joint_names=policy_joint_names,
        default_joint_pos=default_joint_pos,
        default=True,
    )

    # mjswan's _save_web rmtrees output_dir, then copytrees the template,
    # then moves dist/* up. On Windows the inner rmtree of `node_modules`
    # (~4k tiny @tabler icon files) sporadically raises WinError 145.
    # Pre-wipe with a retry-safe rmtree so mjswan's own rmtree is a no-op.
    _robust_rmtree(DIST)

    app = builder.build(output_dir=str(DIST))

    # mjswan's post-build cleanup occasionally fails on Windows, leaving the
    # template's dev-mode index.html (with `<script src="src/index.tsx">`) at
    # the root and an unused src/ subdir. Detect by index.html content (not
    # by the presence of src/, since src/ can survive even when index.html
    # was correctly overwritten). Repair by promoting the built site from
    # the nested dist/ subdir.
    nested_dist = DIST / "dist"
    if nested_dist.exists():
        print(f"[main] mjswan cleanup incomplete; promoting {nested_dist} -> {DIST}")
        for item in nested_dist.iterdir():
            target = DIST / item.name
            if target.exists():
                if target.is_dir():
                    _robust_rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(DIST))
        nested_dist.rmdir()
    for stale in ("src", "node_modules", ".nodeenv", "package.json",
                  "package-lock.json", "tsconfig.json", "vite.config.ts",
                  "eslint.config.cjs", ".browserslistrc"):
        p = DIST / stale
        if p.is_dir():
            _robust_rmtree(p)
        elif p.is_file():
            p.unlink()

    print(f"[main] built {DIST}")

    if not no_launch:
        app.launch(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
