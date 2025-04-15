# 🧠 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

This project implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm using PyTorch. It is specifically built for training two cooperative agents in the Unity ML-Agents **Tennis** environment.

---

## 🚀 Overview

Two agents are trained using:
- Individual actor-critic networks
- Centralized training with decentralized execution
- Replay buffer and soft target network updates
- Ornstein-Uhlenbeck noise for exploration

The goal is to **bounce the ball back and forth** and learn a cooperative policy. The environment is **solved when the average score over 100 episodes is ≥ 0.5**.

---

## Setup

### 1. Install dependencies

```bash
pip install numpy torch matplotlib unityagents

```

Training the Agents
The training loop is wrapped in the maddpg() function:

python
Copy
Edit
scores, rolling_mean = maddpg()
Trains both agents using shared state and experience

Automatically saves models when solved:

checkpoint_actor1_xreplay.pth

checkpoint_critic1_xreplay.pth

checkpoint_actor2_xreplay.pth

checkpoint_critic2_xreplay.pth

### Plotting Results
After training, plot the scores with:

```
plot_result(scores, rolling_mean)
```
This will generate a matplotlib chart showing:

Per-episode scores

Smoothed rolling average (100 episodes)

## Hyperparameters

* BUFFER_SIZE -- 1e6
* BATCH_SIZE -- 512
* GAMMA	-- 0.95
* TAU	-- 0.05
* LR_ACTOR	-- 1e-3
* LR_CRITIC	-- 1e-3
* NOISE_SCALE	-- 1.0
* NOISE_DECAY_LIMIT --	1200
* BETA_EPISODES_LIMIT	-- 3000

