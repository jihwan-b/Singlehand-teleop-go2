# Go2 Convex MPC — Glove Teleoperation

Real-time teleoperation of the Unitree Go2 quadruped via a custom sensor glove,
built on top of a Convex MPC locomotion controller in MuJoCo simulation.

This project was developed as part of the **Yonsei University Human-Centered AI Robotics Lab's 2025 Winter Internship Program**,
and won **1st place** at the Yonsei University VAR Program Final Symposium.

<p align="center">
  <img src="media/VAR Poster.png" width="420">
</p>

---

## Original Repository

This repo is based on [go2-convex-mpc](https://github.com/elijah-waichong-chan/go2-convex-mpc) by **elijah-waichong-chan**.
The original work implements a Convex MPC locomotion controller for the Unitree Go2 quadruped in MuJoCo,
developed as a UC Berkeley MEng capstone project.

---

## What's Added

- **Glove teleoperation** (`teleop/`) — real-time robot control via a custom Arduino glove
  with a magnetometer, haptic motor, and capacitive finger sensors
- **Finger combo → feature mapping** (stand/sit, locomotion, recovery, Euler lean)
- **Demo runner** for quick simulation start

---

## Installation

```bash
git clone https://github.com/elijah-waichong-chan/go2-convex-mpc.git
cd go2-convex-mpc
conda env create -f environment.yml
conda activate go2-convex-mpc
pip install -e .
```

---

## Run

**Glove teleoperation:**
```bash
python teleop/run_teleop.py --port /dev/ttyACM0
```

**Quest3 Remote controller** *(requires [ALVR](https://github.com/alvr-org/ALVR))*:
```bash
python teleop/run_quest3.py
```

**Experiment mode:**
```bash
python teleop/run_experiment.py
```

**Demo With HUD overlay:**
```bash
python teleop/run_demo.py
```

**MPC examples (from original repo):**
```bash
python -m examples.ex00_demo
python -m examples.ex02_trot_forward
```
