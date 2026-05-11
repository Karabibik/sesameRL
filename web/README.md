# Web demo

Browser playback of the trained Sesame policy via [mjswan](https://github.com/ttktjmt/mjswan)
(MuJoCo-WASM + onnxruntime-web).

## Quickstart

From the repo root with the project venv active:

```powershell
python web/sync_policy.py        # copies latest logs/<run>/*.onnx -> web/assets/policy.onnx
python web/main.py               # builds web/dist/ and opens http://localhost:8080
```

Three sliders (forward / lateral / yaw) drive the velocity command.

### Build only

```powershell
python web/main.py --no-launch
```

Or set `MJSWAN_NO_LAUNCH=1`. The static site lands in [dist/](dist/).

### Pin a specific checkpoint

```powershell
python web/sync_policy.py --onnx logs/2026-04-19_BetterWithRewards/2026-04-19_13-21-53.onnx
python web/main.py
```

## How it works

- [main.py](main.py) registers the `Sesame-Velocity-Flat` task by importing
  [play.py](../play.py) (side-effect registration), then uses
  `mjswan.Builder` + `add_mjlab_scene` to pull the MJCF, observations,
  actions, and (most) events directly from the live mjlab env_cfg.
- mjlab's `UniformVelocityCommandCfg` has no auto-mapping in mjswan, so
  [main.py](main.py) substitutes `mjswan.velocity_command(...)` with ranges
  read from `sesame_flat_env_cfg(play=True).commands["twist"].ranges`.
- The ONNX policy is the auto-export mjlab writes alongside every
  PyTorch checkpoint (see `mjlab.rl.runner.MjlabOnPolicyRunner.save`).
  All control metadata (`joint_names`, `default_joint_pos`, `action_scale`,
  `joint_stiffness`, `joint_damping`) is embedded in `metadata_props`.

## Link previews (OG / Twitter Card)

[main.py](main.py) injects custom Open Graph + Twitter Card meta tags
into the built `index.html` so link unfurls (Discord, Slack, iMessage,
Twitter, etc.) show this project's title, description, and a thumbnail
instead of mjswan's defaults.

Edit the `PAGE_TITLE`, `PAGE_DESCRIPTION`, and `PAGE_CANONICAL_URL`
constants near the top of [main.py](main.py) to change the copy.

To add a thumbnail:

1. Take a screenshot (1200×630 ideal — Twitter/Discord crop closer to
   1200×600).
2. Save it as [assets/og.png](assets/og.png).
3. Re-run `python web/main.py`. The build copies the image into
   `web/dist/og.png` and references it in `og:image` /
   `twitter:image`.

If the file is absent, OG tags omit the image and Twitter falls back
to `summary` (small) cards.

After pushing, validate with:
- https://www.opengraph.xyz/url/https%3A%2F%2Fkarabibik.github.io%2FsesameRL%2F
- https://cards-dev.twitter.com/validator

Some social sites (Discord, Twitter) cache previews aggressively. If a
share still shows the old preview, change the URL fragment (e.g.
`?v=2`) to bust the cache.

## Re-export after training

mjlab writes a fresh `<timestamp>.onnx` next to each saved `model_*.pt`.
After a new training run, just re-run `sync_policy.py`:

```powershell
python web/sync_policy.py
python web/main.py
```

## Notes

- **Initial pose** looks askew on first frame — known issue from the
  URDF source ([../README.md](../README.md) roadmap). The policy
  recovers in ~100 ms.
- **Action-scale drift**: ONNX metadata reflects the value the policy
  was *trained* with, which mjswan honors. Editing
  [config.py](../config.py)'s `ACTION_SCALE` does not retroactively
  change the deployed policy — re-train and re-sync.
- **`obs_normalization=False`** ([config.py](../config.py)) is required:
  if turned on, the actor expects normalized inputs that the demo would
  not provide.
- **Solver iterations**: mjlab uses `ccd_iterations=50`
  ([env_cfg.py](../env_cfg.py)); MuJoCo-WASM defaults differ. If the
  browser robot trips while [play.py](../play.py) walks cleanly, override
  the solver settings in `main.py`.
- **Windows**: mjswan boots Node 25 via `nodeenv` on first build; expect
  a one-time ~2-minute install. UTF-8 is required —
  `python web/main.py` sets `PYTHONUTF8=1` is not auto-set. If the build
  crashes on a `UnicodeEncodeError`, run with
  `PYTHONUTF8=1 python web/main.py`.
- **`reset_joints_by_offset` warning** is benign — mjswan has no mapping
  for this mjlab reset event, but it only matters at episode reset
  which the demo doesn't trigger.
