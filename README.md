# sesameRL

RL training environment for the 4-legged, 8-DOF [Sesame quadruped](https://github.com/dorianborian/sesame-robot/), built on [mjlab](https://mujocolab.github.io/mjlab/).

Every observation, reward, event, curriculum stage, and PPO hyperparameter lives in the [config.py](config.py).

---

## Quick start

- Download the repository
- Create and activate a virtual environment
- Install [mjlab](https://mujocolab.github.io/mjlab/)
- Install [PyTorch](https://pytorch.org/) (with gpu)
- From the base folder, with the virtual environment active:

```bash
python validate.py                                          # sinusoidal joint drive, no policy
python train.py                                             # full PPO run
python play.py                                              # auto-loads latest checkpoint
tensorboard --logdir logs                                   # monitor training
```

---

## Web demo
Check the [interactive policy demo](https://karabibik.github.io/sesameRL/)!

---

## Layout

```
sesameRL/
├── validate.py        sanity check
├── config.py          all the configs, main place to edit
├── train.py           run for training
├── play.py            run for sandbox play
├── sesame_robot.py    imports robot from URDF file
├── env_cfg.py         composes mjlab environment from config.py
├── urdf_model/        Sesame.urdf + STL meshes
├── web/               browser demo
└── logs/              TensorBoard + checkpoints + params
```

---

## Robot notes

Link / joint naming inherited from the Sesame project:

| Link    | Role         | Hip joint | Knee joint |
|---------|--------------|-----------|------------|
| Link_L3 | Front-left   | Joint_L1  | Joint_L3   |
| Link_L4 | Rear-left    | Joint_L2  | Joint_L4   |
| Link_R3 | Front-right  | Joint_R1  | Joint_R3   |
| Link_R4 | Rear-right   | Joint_R2  | Joint_R4   |

---

## Network architecture

Both actor and critic share the same MLP shape: two hidden layers of 256 and 128 units with ELU activations.

```
Input (36)  →  256  →  128  →  output
```

The input vector is 36 values in this order:

| Group | Dims | Source |
|---|---|---|
| `base_lin_vel` | 3 | IMU (integrated accel, see note) |
| `base_ang_vel` | 3 | IMU gyroscope |
| `projected_gravity` | 3 | IMU orientation |
| `joint_pos` | 8 | joint encoders (delta from default pose) |
| `joint_vel` | 8 | joint encoders (velocity) |
| `actions` | 8 | previous output — stored in software |
| `command` (vx, vy, wz) | 3 | user / higher-level controller |

Output is 8 joint position targets (one per joint).  
The critic (output size 1) is **discarded after training** — only the actor is needed on the robot.

---

## Sim-to-real / hardware notes (ESP32S3 + MG90S)

### What the deployed policy needs

| Observation | What provides it | Notes |
|---|---|---|
| `projected_gravity` | MPU6050 DMP quaternion | Rotate [0, 0, −g] into body frame and normalise |
| `base_ang_vel` | MPU6050 DMP gyro | Directly available |
| `base_lin_vel` | MPU6050 accelerometer + integration | **Drifts**; zeroing it or clamping the integral is a reasonable first approximation |
| `joint_pos` | MG90S (open-loop) | No feedback wire — use the last *commanded* position as the estimate |
| `joint_vel` | Derived | Finite-difference of commanded position; noisy but avoids extra hardware |
| `actions` | Software buffer | Store last output on the MCU — no sensor needed |
| `command` | Firmware / RC input | Set by your controller — no sensor needed |

### What is only needed during training

- **Critic network** — entirely dropped; never runs on the robot.
- **Foot contact / foot-slip signal** — used only for the `foot_slip` penalty reward. The deployed actor has no foot contact inputs and does not need contact sensors.
- **Ground-truth base linear velocity** — sim provides this for free. On hardware it is estimated (see above).
- **Terrain height samples** — not part of the observation vector; not needed even for rough-terrain policies.

### MG90S servo timing

The default firmware pulses each of the 8 servos sequentially with a 40 ms step, giving a full-cycle latency of ~320 ms (≈ 3 Hz). The sim control loop runs considerably faster. Before deploying a trained policy, this gap will need to be addressed — either by reducing per-servo delay, broadcasting commands in parallel over a serial bus, or adding latency-compensation to the observations.

---

## Roadmap

- [ ] URDF checks
	- [ ] Solidworks->URDF is not very mature, so some positions/axes might be off
	- [ ] Starting orientation is garbage visually. Has no adverse effect, but would be nice if it was standing straight.
- [ ] Check the [TODO] and [OPT] tags in [config.py](config.py)
    - [ ] Tune kp-kd values
    - [x] Make the NN small enough to fit ESP32
    - [ ] Current observations are not realizable with existing hardware
    - [ ] Currently uses curriculum, can upgrade to hierarchical learning
    - [x] Play with rewards for better locomotion
- [x] Terrain is flat, can upgrade to a rough terrain
- [ ] Sim-to-real preparation (needs extended hardware)
- [x] Add a web-viewer using [mjswan](https://github.com/ttktjmt/mjswan)
- [x] Resume training from an existing checkpoint