# UAV Navigation with PPO

Reinforcement learning agent trained to navigate a quadrotor through waypoints in a 3D flight environment, built as **Aşama 2** of a defense-industry-oriented portfolio centered on the theme: *"engineer who perceives (CV) + decides (RL) + measures reliability in both."*

Aşama 1 (`rl-fundamentals`) covered PPO/DQN fundamentals from scratch on CartPole. This project applies those fundamentals to a real navigation task using a production-grade RL library, and investigates whether the trained policy's reliability can be measured — laying groundwork for Aşama 3 (policy uncertainty quantification).

## Environment

- **Simulator:** [PyFlyt](https://github.com/jjshoots/PyFlyt) `QuadX-Waypoints-v4` — a PyBullet-based quadrotor simulation
- **Task:** navigate to up to 4 sequential waypoints in open space; episode ends on reaching all waypoints, colliding with the ground, exiting the flight boundary, or timing out
- **Observation:** flattened to a 24-dim vector — 21-dim drone attitude (angular velocity, quaternion, linear velocity, position, previous action, motor state) + 3-dim delta to the next waypoint
- **Action:** 4-dim continuous — roll/pitch/yaw rate setpoints `[-π, π]` and thrust `[0, 0.8]`

## Approach

A hand-written PPO implementation (validated on CartPole in Aşama 1) was tried first but failed to learn in this environment — reward stayed flat, entropy never decreased even after 200k timesteps. Rather than debug a custom implementation indefinitely, training moved to **Stable-Baselines3**, which learned the task successfully under identical conditions. This is a deliberate engineering choice: Aşama 1 already demonstrated PPO understood from first principles; this project's value lies in the navigation task and the reliability analysis on top of it, not in re-proving the algorithm.

Training was done in two stages:
1. **200k timesteps** — sanity check, confirmed learning was occurring
2. **1M timesteps** — final model, used for all reported results

## Results

Evaluated over 20 fixed seeds, deterministic policy:

| Metric | Value |
|---|---|
| Mean targets reached | 3.00 / 4 |
| Episodes reaching all 4 targets | 9 / 20 (45%) |
| Collision rate | 3 / 20 (15%) |
| Out-of-bounds rate | 1 / 20 (5%) |
| Mean episode reward | 438.6 ± 189.2 |

Failure analysis (per-seed breakdown) showed that all collisions were concentrated in 3 specific seeds, each ending in a smooth, controlled — but uncorrected — descent straight into the ground, rather than random loss of control. This finding shaped the reliability experiments below.

## Reliability signature attempts (negative results, documented honestly)

Following the CV-side reliability signature (ECE + confidence thresholding + OOD flagging), two approaches were tried to give the RL agent an analogous "know when you might fail" signal. Both are reported here because the negative result and its diagnosis are informative in their own right.

**1. Altitude-based reward shaping.** Hypothesis: penalizing low altitude would discourage the uncorrected descents causing collisions. Result: targets-reached dropped from 3.00 to 2.70 and collision rate did not improve. Diagnosis: the penalty targeted the wrong variable. The crash trajectories showed a smooth, monotonic altitude decrease from normal cruising height all the way to the ground — the problem was the agent's inability to arrest its *descent rate* near the ground, not its raw altitude. Penalizing altitude broadly discouraged legitimate low-altitude approaches to waypoints, degrading overall performance without fixing the actual failure mode.

**2. MC Dropout epistemic uncertainty.** Hypothesis: injecting dropout into the trained policy and measuring the variance of repeated stochastic forward passes at a given state would surface elevated uncertainty before failures. Result: mean uncertainty for collision episodes (≈0.50) was statistically indistinguishable from non-collision episodes (≈0.48) — within noise. Diagnosis: PPO converges to a highly deterministic, low-entropy policy; small weight perturbations via dropout do not meaningfully perturb its output at the states that matter. This is a known tension between MC Dropout (designed for supervised/Bayesian settings) and on-policy RL, where the policy is trained to be sharp rather than calibrated.

Both attempts are left as documented dead ends rather than removed, since the diagnostic process — not just the result — is the point.

## Repository structure

```
uav_navigation_ppo.ipynb    # full training + evaluation notebook (Colab)
ppo_quadx_waypoints_1M.zip  # final trained model (Stable-Baselines3 format)
requirements.txt
README.md
```

## Reproducing

```bash
pip install -r requirements.txt
```

```python
from stable_baselines3 import PPO
model = PPO.load("ppo_quadx_waypoints_1M")
```

See the notebook for environment wrappers (`FlattenWaypointObs`, `ClipAction`) required to reconstruct the training/eval environment.

## Next steps (Aşama 3)

The two failed reliability attempts point toward **ensemble disagreement** as the next candidate: training several independently-seeded PPO models and using their action disagreement at a given state as an uncertainty proxy. This is more expensive (multiple full training runs) but avoids the structural issues found above, since it doesn't rely on perturbing a single deterministic policy.
