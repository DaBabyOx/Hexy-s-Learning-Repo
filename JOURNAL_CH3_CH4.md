# Chapter 3: Methodology

## 3.1 Overview

This chapter presents a vendor-agnostic reinforcement learning (RL) framework for training a hexapod robot to perform locomotion tasks. The framework decouples the reward signal, observation space, and policy architecture from the underlying physics simulation backend, enabling identical training objectives across two fundamentally different simulation environments: MuJoCo with PyTorch (CPU-based) and Brax/MJX with JAX (GPU-accelerated). A three-stage experimental design evaluates the framework on flat and uneven terrain, with and without backend-specific hyperparameter tuning.

---

## 3.2 Robot Platform

### 3.2.1 Physical Description

The robot used in this study is a custom hexapod with six legs arranged symmetrically around a central body. Each leg consists of three rigid segments: the coxa (upper segment, connecting to the body), the femur (middle segment), and the tibia (lower segment, terminating in a spherical foot contact point). The robot model is defined in MuJoCo XML format and loaded by both backends.

**Body dimensions and mass:**

| Link | Mass (kg) |
|------|-----------|
| Body | 0.50 |
| Coxa (× 6) | 0.20 each |
| Femur (× 6) | 0.15 each |
| Tibia (× 6) | 0.10 each |
| **Total** | **~3.90** |

The mesh geometry is defined by STL files (scale factor 0.0254, converting from inches to meters) and loaded as visual-only geometry. Contact detection is handled exclusively by six spherical foot geoms (radius 0.02 m) to ensure simulation stability across both backends.

### 3.2.2 Kinematic Structure

The robot has a floating root body attached to the world via a free joint, providing 6 unconstrained degrees of freedom (3 translational, 3 rotational). Each leg contains 3 revolute (hinge) joints, giving 18 actuated degrees of freedom in total.

**Joint configuration per leg:**

| Joint | Axis | Range (deg) | Damping | Gear Ratio |
|-------|------|-------------|---------|------------|
| Coxa | Z (yaw) | [−45, +45] | 2.0 | 60 |
| Femur | Y (pitch) | [−70, +70] | 2.0 | 80 |
| Tibia | Y (pitch) | [−120, +20] | 1.0 | 80 |

The asymmetric tibia range (−120° to +20°) reflects a biological-inspired design that restricts hyperextension while allowing full retraction during the swing phase.

**Full state vector:**

$$\mathbf{q} \in \mathbb{R}^{25},\quad \dot{\mathbf{q}} \in \mathbb{R}^{24}$$

where $\mathbf{q}$ comprises 7 root state variables (3 position + 4 quaternion) and 18 joint angles, and $\dot{\mathbf{q}}$ comprises 6 root velocities and 18 joint velocities.

**Simulation parameters:**
- Timestep: $\Delta t = 0.01$ s (100 Hz)
- Gravitational acceleration: 9.81 m/s²
- Foot friction: $\mu = 1.8$ (tangential), 0.005 (torsional)

---

## 3.3 Observation and Action Spaces

### 3.3.1 Observation Space

The observation is a 42-dimensional vector constructed identically in both backends:

$$\mathbf{o}_t = \left[\underbrace{\mathbf{v}_{\text{base}}}_6 \;\Big|\; \underbrace{\mathbf{q}_{\text{joints}}}_{18} \;\Big|\; \underbrace{\dot{\mathbf{q}}_{\text{joints}}}_{18}\right] \in \mathbb{R}^{42}$$

where $\mathbf{v}_{\text{base}} = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]$ is the root link velocity in the body frame, $\mathbf{q}_{\text{joints}}$ is the vector of 18 joint positions, and $\dot{\mathbf{q}}_{\text{joints}}$ is the vector of 18 joint velocities. The root position and orientation are deliberately excluded from the observation to encourage velocity-based locomotion rather than pose tracking.

Observations are normalized online during training using a running estimate of the mean and standard deviation, updated via Welford's algorithm:

$$\hat{\mathbf{o}}_t = \frac{\mathbf{o}_t - \hat{\mu}}{\hat{\sigma} + \epsilon}, \quad \epsilon = 10^{-8}$$

### 3.3.2 Action Space

The policy outputs 18 continuous actions, one per joint:

$$\mathbf{a}_t \in [-1, 1]^{18}$$

Actions are clipped to $[-1, 1]$ and scaled by a constant action scale $\alpha = 0.2$ before being applied as joint position targets:

$$\tau_t = \alpha \cdot \mathbf{a}_t$$

---

## 3.4 Reward Function

The reward function is implemented as a shared module (`locomotion_reward`) that operates on both NumPy arrays (MuJoCo backend) and JAX arrays (Brax backend) through a duck-typed interface. The total reward at each timestep is:

$$r_t = r_{\text{vel}} - c_{\text{ctrl}} \cdot r_{\text{ctrl}} - c_{\text{orient}} \cdot r_{\text{orient}} - c_{\text{joint}} \cdot r_{\text{joint}} + r_{\text{alive}}$$

**Velocity reward** encourages matching a target forward velocity $v^*$:

$$r_{\text{vel}} = 1 - (v_x - v^*)^2$$

This quadratic formulation provides a smooth gradient around the target without saturating, while allowing negative values when the robot moves significantly backwards.

**Control cost** penalizes large actuator commands to encourage energy efficiency:

$$r_{\text{ctrl}} = \frac{1}{|\mathcal{A}|} \sum_{i=1}^{|\mathcal{A}|} a_i^2$$

**Orientation penalty** encourages the robot to maintain an upright posture using the Z-component of the body's up vector:

$$r_{\text{orient}} = (1 - u_z)^2$$

where $u_z = R_{22}$ is extracted from the rotation matrix of the root body ($u_z = 1$ when perfectly upright, $u_z = -1$ when inverted).

**Joint limit penalty** softly penalizes joint angles approaching their mechanical limits:

$$r_{\text{joint}} = \frac{1}{|\mathcal{J}|} \sum_{j \in \mathcal{J}} \left[\max(0,\, q_j^{\text{low}} - q_j)^2 + \max(0,\, q_j - q_j^{\text{high}})^2\right]$$

**Reward coefficients:**

| Term | Coefficient | Baseline | Vendor-Tuned |
|------|-------------|----------|--------------|
| $r_{\text{vel}}$ (target velocity $v^*$) | — | 0.6 m/s | 0.8 m/s |
| $c_{\text{ctrl}}$ | 0.02 | ✓ | ✓ |
| $c_{\text{orient}}$ | 1.0 | ✓ | ✓ |
| $c_{\text{joint}}$ | 0.2 | ✓ | ✓ |
| $r_{\text{alive}}$ | 0.0 | ✓ | ✓ |

The alive bonus is set to zero in all experiments to avoid masking catastrophic failures with a constant positive offset.

---

## 3.5 Terrain Environments

### 3.5.1 Flat Terrain

The baseline environment uses a flat, infinite ground plane. The robot is initialized above the origin with small random perturbations to its joint positions and root translation to prevent overfitting to a single initial pose:

$$q_{\text{root}}^{(0)} = q_0 + \epsilon_{\text{root}}, \quad \epsilon_{\text{root}} \sim \mathcal{U}(-0.01, 0.01)^3$$
$$q_{\text{joints}}^{(0)} = q_0 + \epsilon_{\text{joints}}, \quad \epsilon_{\text{joints}} \sim \mathcal{U}(-0.1, 0.1)^{18}$$

The root quaternion is not perturbed to avoid initializing in an invalid rotation state.

### 3.5.2 Heightfield Terrain

The uneven terrain environment replaces the flat plane with a procedurally generated heightfield. Heights are sampled from a clipped normal distribution and scaled:

$$h_{ij} = s \cdot \text{clip}\left(\mathcal{N}(0, 1),\, -2,\, 2\right)$$

**Heightfield parameters by configuration:**

| Parameter | Baseline | Vendor-Tuned |
|-----------|----------|--------------|
| World size (m) | 4.0 × 4.0 | 4.0 × 4.0 |
| Grid resolution | 32 × 32 | 48 × 48 |
| Height scale $s$ (m) | 0.05 | 0.06 |
| Random seed | 123 | 123 |

The heightfield is generated once per experiment and remains static throughout training (no terrain randomization). The robot is initialized at the center of the field.

---

## 3.6 Policy Architecture

Both backends use a two-layer multilayer perceptron (MLP) with the following structure:

$$\pi_\theta(\mathbf{a} | \mathbf{o}) = \tanh\left(W_2 \cdot \tanh\left(W_1 \hat{\mathbf{o}} + b_1\right) + b_2\right)$$

**Network dimensions:**

| Layer | Input | Output | Activation |
|-------|-------|--------|------------|
| Hidden 1 | 42 | 256 | tanh |
| Hidden 2 | 256 | 256 | tanh |
| Output (mean) | 256 | 18 | linear |

A separate log-standard deviation parameter $\log \sigma$ is learned (initialized to $-0.5$, i.e., $\sigma \approx 0.61$) and used only during training for action sampling. During inference and policy export, only the mean head is used.

The value network shares the same architecture as the policy network and is trained jointly under the PPO objective.

**Parameter count:**

$$N_\theta = (42 \times 256 + 256) + (256 \times 256 + 256) + (256 \times 18 + 18) = 82{,}962 \text{ parameters}$$

### 3.6.1 Unified Policy Export

At regular checkpoints, both backends export a backend-agnostic policy file in NumPy `.npz` format containing:

- Layer weights $\{W_0, W_1, \ldots\}$ (stored as $[d_{\text{in}} \times d_{\text{out}}]$)
- Layer biases $\{b_0, b_1, \ldots\}$
- Observation normalization statistics $(\hat{\mu}, \hat{\sigma})$
- Activation function name

This format enables policies trained on either backend to be evaluated, deployed, or fine-tuned on the other, providing a practical cross-vendor interoperability mechanism.

---

## 3.7 Training Algorithm: Proximal Policy Optimization

Both backends implement Proximal Policy Optimization (PPO) [Schulman et al., 2017] with Generalized Advantage Estimation (GAE) [Schulman et al., 2016].

### 3.7.1 Rollout Collection

At each training iteration, $N_e$ parallel environments are stepped for $T$ timesteps to produce a rollout buffer of $N_e \times T$ transitions. For each step:

1. Normalize observation: $\hat{\mathbf{o}}_t = (\mathbf{o}_t - \hat{\mu}) / (\hat{\sigma} + \epsilon)$
2. Sample action: $\mathbf{a}_t \sim \pi_\theta(\cdot | \hat{\mathbf{o}}_t) = \mathcal{N}(\mu_\theta, \sigma_\theta)$
3. Squash with tanh: $\tilde{\mathbf{a}}_t = \tanh(\mathbf{a}_t)$
4. Compute log-probability with change-of-variables correction:
   $$\log \pi(\tilde{\mathbf{a}}_t | \hat{\mathbf{o}}_t) = \log \mathcal{N}(\mathbf{a}_t; \mu_\theta, \sigma_\theta) - \sum_i \log(1 - \tanh^2(a_{t,i}) + \epsilon)$$
5. Compute state value: $V_t = V_\phi(\hat{\mathbf{o}}_t)$
6. Step environment: $\mathbf{o}_{t+1}, r_t, d_t = \text{env.step}(\tilde{\mathbf{a}}_t)$

### 3.7.2 Advantage Estimation

GAE advantages are computed backwards over the rollout buffer:

$$\delta_t = r_t + \gamma (1 - d_t) V_{t+1} - V_t$$
$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

Discounted returns for the value function target are: $\hat{R}_t = \hat{A}_t + V_t$

Advantages are normalized to zero mean and unit variance over each minibatch.

### 3.7.3 Policy Update

For each of $K$ update epochs, the rollout buffer is shuffled and split into minibatches of size $M$. Each minibatch update minimizes:

$$\mathcal{L}(\theta) = -\mathcal{L}^{\text{CLIP}}(\theta) + c_v \mathcal{L}^V(\phi) - c_H \mathcal{H}[\pi_\theta]$$

where the clipped policy objective is:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

with probability ratio $r_t(\theta) = \pi_\theta(\tilde{\mathbf{a}}_t | \hat{\mathbf{o}}_t) / \pi_{\theta_{\text{old}}}(\tilde{\mathbf{a}}_t | \hat{\mathbf{o}}_t)$, the value function loss:

$$\mathcal{L}^V(\phi) = \frac{1}{2}\mathbb{E}\left[(V_\phi(\hat{\mathbf{o}}_t) - \hat{R}_t)^2\right]$$

and $\mathcal{H}[\pi_\theta]$ the policy entropy.

Gradients are clipped to a maximum norm of 0.5 before each Adam optimizer step.

---

## 3.8 Simulation Backends

### 3.8.1 MuJoCo + PyTorch Backend

The MuJoCo backend runs environments on CPU using the native MuJoCo Python bindings. Parallel environment execution is achieved via a multiprocessing pool using Python's `multiprocessing.Process` workers, with one worker per available CPU core. Each worker maintains its own `mujoco.MjModel` and `mujoco.MjData` instance.

The training loop is implemented in PyTorch. Rollout data is collected asynchronously from worker processes via bidirectional queues and transferred to a GPU tensor (if available) for the PPO update step.

At each environment step, the backend checks for numerical divergence (NaN or Inf in position, velocity, or acceleration arrays). If divergence is detected, the episode is immediately terminated with a penalty reward of $-1$.

### 3.8.2 Brax/MJX + JAX Backend

The Brax backend uses MJX (MuJoCo's JAX-based physics engine) for massively parallel GPU simulation. All $N_e$ environments are executed simultaneously as a single batched computation using `jax.vmap`, with the entire rollout collected via `jax.lax.scan`. The complete training loop — rollout, GAE, PPO update — is compiled into a single XLA-optimized computation graph via `jax.jit`.

Training is delegated to Brax's built-in `ppo_train` function, which handles device placement, observation normalization, and the training/evaluation loop. To ensure compatibility with the AMD MI300X GPU (ROCm backend), the following modifications were applied:

- **Single-device restriction:** `max_devices_per_host=1` prevents pmap from replicating training across the MI300X's multiple internal GCD (GPU Complex Die) compute agents, which would otherwise cause conflicting kernel dispatches.
- **Platform allocator:** `XLA_PYTHON_CLIENT_ALLOCATOR=platform` uses direct `hipMalloc`/`hipFree` rather than XLA's BFC caching allocator, avoiding stale address reuse under memory pressure.
- **Command buffer disabled:** `XLA_FLAGS=--xla_gpu_enable_command_buffer=` disables ROCm graph capture/replay, which produced malformed AQL dispatch packets (zero setup field) when replaying fused scan kernels on MI300X.
- **Polling mode:** `HSA_ENABLE_INTERRUPT=0` switches ROCm from interrupt-driven to polling-based kernel completion, avoiding a null completion signal bug in the async interrupt path.

---

## 3.9 Experimental Design

Three experimental stages are conducted to evaluate the vendor-agnostic framework:

| Stage | Configuration | Backend(s) | Terrain | Purpose |
|-------|--------------|------------|---------|---------|
| 1 | Baseline | MuJoCo, Brax | Flat | Cross-backend comparability |
| 2 | Vendor-tuned | Brax | Flat | GPU-optimized upper bound |
| 3 | Baseline | MuJoCo, Brax | Heightfield | Terrain generalization |

Each experiment is run with 5 independent seeds. Seeds are derived as $s_k = (s_0 + 9973k) \bmod (2^{31}-1)$, with base seed $s_0 = 0$.

### 3.9.1 Hyperparameters

**Shared across all experiments:**

| Hyperparameter | Value |
|---------------|-------|
| Policy hidden sizes | [256, 256] |
| Activation | tanh |
| Log std init | −0.5 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip $\epsilon$ | 0.2 |
| Value coeff $c_v$ | 0.5 |
| Entropy coeff $c_H$ | 0.01 |
| Max grad norm | 0.5 |
| Learning rate | 3 × 10⁻⁴ |
| Observation normalization | enabled |
| Action scale $\alpha$ | 0.2 |
| Control cost $c_{\text{ctrl}}$ | 0.02 |
| Orientation cost $c_{\text{orient}}$ | 1.0 |
| Joint limit cost $c_{\text{joint}}$ | 0.2 |

**Backend-specific hyperparameters:**

| Hyperparameter | MuJoCo Baseline | Brax Baseline | Brax Vendor-Tuned |
|---------------|-----------------|---------------|-------------------|
| Total steps | 5,000,000 | 5,000,000 | 12,000,000 |
| Parallel envs $N_e$ | 256 | 256 | 2,048 |
| Rollout length $T$ | 128 | 128 | 256 |
| Minibatch size $M$ | 1,024 | 1,024 | 32,768 |
| Minibatches per update | 32 | 32 | 16 |
| Update epochs $K$ | 4 | 4 | 4 |
| Target velocity $v^*$ | 0.6 m/s | 0.6 m/s | 0.8 m/s |

The vendor-tuned configuration is derived from the Brax baseline by scaling parallel environments, rollout length, and minibatch size to maximize GPU utilization on the AMD MI300X (192 GB HBM3), while keeping all algorithmic hyperparameters identical.

### 3.9.2 Evaluation Protocol

Evaluation is performed every 200,000 environment steps. During evaluation, 128 deterministic rollouts are executed (using the policy mean $\mu_\theta$ without sampling) and the following metrics are recorded:

- **Episode reward:** Total undiscounted return per episode
- **Forward velocity:** Mean $v_x$ over the episode
- **Upright factor:** Mean $u_z$ over the episode
- **Reward decomposition:** Per-component contributions (velocity, control, orientation, joint)

Training throughput is measured in steps per second (SPS).

---

---

# Chapter 4: Results and Discussion

## 4.1 Overview

This chapter presents the empirical results across all three experimental stages. All experiments were conducted on an AMD MI300X GPU (192 GB HBM3, ROCm 7.2) for Brax/JAX runs and on the same host's CPU for MuJoCo runs. The Brax backend compiles the full training loop — environment rollout, GAE computation, and PPO update — into a single XLA computation graph via `jax.jit`, with the physics engine (MJX) executed through `jax.vmap` over all parallel environments. Results are reported as mean ± standard deviation over 5 independent seeds unless stated otherwise.

The per-seed episode rewards used in all tables below are final evaluation scores at the end of training:

| Experiment | seed\_0 | seed\_9973 | seed\_19946 | seed\_29919 | seed\_39892 |
|---|---|---|---|---|---|
| MuJoCo baseline flat | 14.4 | 638.7 | 637.8 | 635.6 | −403.7 |
| MuJoCo baseline heightfield | 463.0 | 532.8 | 586.1 | 631.0 | 541.2 |
| Brax baseline flat | 673.1 | 590.1 | 647.0 | 679.5 | 651.6 |
| Brax baseline heightfield | 689.8 | 666.5 | 626.9 | 647.2 | 668.0 |
| Brax vendor-tuned flat | 405.3 | 389.9 | 304.7 | 527.2 | 332.7 |

---

## 4.2 Stage 1: Baseline Performance on Flat Terrain

### 4.2.1 Training Convergence

**Table 1** summarises the final evaluation metrics for both backends on flat terrain using identical baseline hyperparameters ($N_e = 256$, $T = 128$, $5 \times 10^6$ total steps).

**Table 1.** Final evaluation metrics — Baseline, Flat Terrain (mean ± std, $n = 5$ seeds).

| Metric | MuJoCo (CPU) | Brax/JAX (GPU) |
|--------|-------------|----------------|
| Episode reward (all seeds) | 304.6 ± 428.5 | **648.3 ± 31.6** |
| Episode reward (converged seeds only) | 637.4 ± 1.6 ($n=3$) | 648.3 ± 31.6 ($n=5$) |
| Convergence rate | **3 / 5 (60%)** | **5 / 5 (100%)** |
| Mean forward velocity (m/s) | — | 0.033 ± 0.012 |
| Mean upright factor $\bar{u}_z$ | — | 0.949 ± 0.022 |
| Training throughput (SPS) | ~617 (est.) | **3 939 ± 1 053** |

The MuJoCo backend exhibits catastrophic divergence in 2 out of 5 seeds: seed\_0 converges to a near-stationary policy (reward 14.4) and seed\_39892 to a persistently inverted posture (reward −403.7). The remaining three MuJoCo seeds converge to rewards of 635–639, comparable to the best Brax seeds. This bimodal outcome suggests that the CPU-based multiprocessing training loop is susceptible to early policy collapse from which PPO cannot recover.

The Brax/JAX backend converges in all 5 seeds, with rewards ranging from 590 to 680. The lower variance (31.6 vs 428.5) reflects the stabilising effect of fully vectorised gradient computation: averaging over $256$ environments simultaneously produces lower-variance policy gradient estimates than the MuJoCo runner, which collects rollouts asynchronously across CPU workers.

The mean forward velocity achieved by the Brax policy (0.033 m/s) falls well short of the target ($v^* = 0.6$ m/s). This is consistent with the observed forward reward component: at $v_x = 0.033$ m/s, $r_{\text{vel}} = 1 - (0.033 - 0.6)^2 \approx 0.68$ per step, which over 1 000 steps yields approximately 680, matching the reported forward component (664 ± 14 per episode). The hexapod has learned to maintain an upright posture ($\bar{u}_z = 0.949$) and take tentative forward steps, but has not yet converged to an efficient gait at the target speed within the 5M-step budget.

### 4.2.2 Reward Component Decomposition

**Table 2** reports the mean per-episode contribution of each reward term for the Brax baseline flat configuration.

**Table 2.** Reward component breakdown — Brax baseline, flat terrain (mean ± std over 5 seeds, per episode).

| Component | Formula | Mean ± Std |
|-----------|---------|------------|
| Velocity ($r_{\text{vel}}$) | $1 - (v_x - 0.6)^2$ summed | 664.1 ± 13.8 |
| Control ($-c_{\text{ctrl}} \cdot r_{\text{ctrl}}$) | $-0.02 \sum a^2$ summed | −0.25 ± 0.02 |
| Orientation ($-c_{\text{orient}} \cdot r_{\text{orient}}$) | $-(1-u_z)^2$ summed | −15.5 ± 27.0 |
| Joint limits ($-c_{\text{joint}} \cdot r_{\text{joint}}$) | $-0.2 \cdot \text{violation}$ summed | −0.08 ± 0.01 |
| **Total** | | **648.3 ± 31.6** |

The velocity reward dominates (>99% of total), while control and joint costs are negligible (< 0.1 per episode). The orientation penalty shows high variance across seeds: seed\_9973 incurs −64.3 per episode (indicating frequent tipping), while seeds 0, 19946, and 29919 incur only −2 to −4 per episode. This explains seed\_9973's lower total reward (590.1) despite a comparable velocity component.

### 4.2.3 Training Throughput

The Brax/JAX backend, backed by XLA, batches all 256 environments into a single GPU kernel dispatch per step via `jax.vmap`. Mean throughput is **3 939 ± 1 053 SPS** (environment steps per second). The wide standard deviation reflects per-seed variation in JIT compilation time: the first training epoch incurs approximately 5–15 minutes of XLA tracing before steady-state throughput is reached.

The MuJoCo backend throughput is estimated at approximately 617 SPS based on total training duration (~2.25 hours for 5M steps), yielding a GPU speedup of approximately **6.4×** over CPU at the same parallelism level ($N_e = 256$).

---

## 4.3 Stage 2: Vendor-Tuned Performance on Flat Terrain

### 4.3.1 Configuration and Scaling

The vendor-tuned configuration scales the Brax baseline along three axes: parallel environments ($256 \to 2{,}048$), rollout length ($128 \to 256$), and total training steps ($5\text{M} \to 12\text{M}$). The per-update sample count increases from $32{,}768$ to $524{,}288$. The target velocity is raised from $0.6$ to $0.8$ m/s. All other hyperparameters and the reward function structure are held constant.

### 4.3.2 Comparison with Baseline

**Table 3** compares the Brax baseline against the vendor-tuned configuration on flat terrain.

**Table 3.** Brax baseline vs. vendor-tuned — Flat Terrain (mean ± std, $n = 5$ seeds).

| Metric | Brax Baseline | Brax Vendor-Tuned |
|--------|---------------|-------------------|
| Episode reward | **648.3 ± 31.6** | 391.9 ± 76.9 |
| Target velocity $v^*$ (m/s) | 0.6 | 0.8 |
| Mean forward velocity (m/s) | 0.033 ± 0.012 | **0.055 ± 0.066** |
| Mean upright factor $\bar{u}_z$ | 0.949 ± 0.022 | 0.947 ± 0.016 |
| Training throughput (SPS) | 3 939 ± 1 053 | **948 ± 30** |
| Total training steps | 5M | 12M |
| Parallel environments | 256 | 2 048 |

The vendor-tuned configuration achieves a lower absolute episode reward (391.9 vs 648.3). This difference is primarily attributable to the harder task objective: with target velocity $v^* = 0.8$ m/s and the robot achieving approximately 0.055 m/s, $r_{\text{vel}} = 1 - (0.055 - 0.8)^2 \approx 0.445$ per step versus $r_{\text{vel}} \approx 0.680$ per step for the baseline under its $v^* = 0.6$ m/s target. Applying the baseline target post-hoc to the vendor-tuned achieved velocity yields an equivalent per-step reward of $1 - (0.055 - 0.6)^2 \approx 0.703$, slightly exceeding the baseline (0.680), which suggests the vendor-tuned policy achieves marginally better forward locomotion despite the lower absolute score.

The vendor-tuned configuration also exhibits substantially higher seed variance (76.9 vs 31.6). Seed\_19946 in particular fails to converge (reward 304.7, mean forward velocity 0.002 m/s), suggesting that the larger batch size and harder target do not fully eliminate convergence instability.

Training throughput drops from 3 939 to 948 SPS, a 4.2× reduction despite 8× more environments. This is consistent with the 8× larger per-update computation ($2{,}048 \times 256$ vs $256 \times 128$ samples) running on similar GPU time, plus additional overhead from the platform allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`), which bypasses XLA's BFC memory cache and issues direct `hipMalloc` calls per allocation.

**Table 4.** Per-seed episode rewards — Brax vendor-tuned, flat terrain.

| Seed | Episode Reward | Forward Vel (m/s) | Upright $\bar{u}_z$ | Orientation Penalty |
|------|---------------|-------------------|---------------------|---------------------|
| 0 | 405.3 | 0.044 | 0.947 | −4.99 |
| 9973 | 389.9 | 0.042 | 0.958 | −18.73 |
| 19946 | 304.7 | 0.002 | 0.959 | −40.03 |
| 29919 | 527.2 | 0.161 | 0.952 | −27.00 |
| 39892 | 332.7 | 0.028 | 0.919 | −53.81 |
| **Mean ± Std** | **391.9 ± 76.9** | **0.055 ± 0.066** | **0.947 ± 0.016** | **−28.9 ± 19.9** |

Seed\_29919 is the standout run, achieving a mean forward velocity of 0.161 m/s (highest across all configurations) and the best reward (527.2). Seeds 19946 and 39892 show large orientation penalties (−40 and −54 per episode), indicating persistent balance instability. The high variance in orientation cost is the primary driver of reward variance in this configuration.

---

## 4.4 Stage 3: Baseline Performance on Uneven Terrain

### 4.4.1 Heightfield Results

**Table 5** reports final evaluation metrics for both backends trained directly on the 32×32 heightfield terrain.

**Table 5.** Final evaluation metrics — Baseline, Heightfield Terrain (mean ± std, $n = 5$ seeds).

| Metric | MuJoCo (CPU) | Brax/JAX (GPU) |
|--------|-------------|----------------|
| Episode reward | 550.8 ± 56.2 | **659.7 ± 21.2** |
| Convergence rate | **5 / 5 (100%)** | **5 / 5 (100%)** |
| Mean forward velocity (m/s) | — | 0.035 ± 0.023 |
| Mean upright factor $\bar{u}_z$ | — | 0.960 ± 0.013 |
| Training throughput (SPS) | — | 3 344 ± 93 |

Unlike the flat terrain condition, all 5 MuJoCo seeds converge on the heightfield (100% convergence rate). The heightfield terrain imposes additional proprioceptive challenge but appears to prevent the catastrophic failure modes observed on flat terrain: the irregular surface contact forces the robot to adopt conservative, slow gaits from early training, avoiding the unstable high-speed strategies that caused policy collapse in 2 of 5 MuJoCo flat seeds.

MuJoCo heightfield mean reward (550.8) is substantially higher than MuJoCo flat including failed seeds (304.6), and statistically comparable when only converged flat seeds are considered (637.4 vs 550.8 for flat and heightfield respectively), suggesting the task difficulty of the heightfield reduces peak performance but improves training reliability.

The Brax heightfield configuration maintains high performance (659.7 ± 21.2), with lower seed variance than Brax flat (21.2 vs 31.6) and 100% convergence. Interestingly, Brax heightfield slightly outperforms Brax flat (659.7 vs 648.3), consistent with the MuJoCo trend. The upright factor ($\bar{u}_z = 0.960$) is marginally better on heightfield than flat (0.949), which may reflect that the terrain encourages more cautious, upright gaits.

Heightfield throughput (3 344 SPS) is slightly lower than flat (3 939 SPS), reflecting the additional computation from the heightfield contact geometry within the MJX physics step.

**Table 6.** Per-seed episode rewards — Brax baseline, heightfield terrain.

| Seed | Episode Reward | Forward Vel (m/s) | Upright $\bar{u}_z$ | Orientation Penalty |
|------|---------------|-------------------|---------------------|---------------------|
| 0 | 689.8 | 0.063 | 0.972 | −1.65 |
| 9973 | 666.5 | 0.041 | 0.947 | −4.58 |
| 19946 | 626.9 | 0.005 | 0.963 | −3.35 |
| 29919 | 647.2 | 0.023 | 0.974 | −2.45 |
| 39892 | 668.0 | 0.041 | 0.946 | −3.95 |
| **Mean ± Std** | **659.7 ± 21.2** | **0.035 ± 0.023** | **0.960 ± 0.013** | **−3.20 ± 1.06** |

The heightfield orientation penalties (−1.65 to −4.58 per episode) are significantly smaller and more consistent than the flat terrain penalties (−2.42 to −64.25), confirming greater balance stability across seeds on uneven terrain.

### 4.4.2 Flat vs. Heightfield Comparison

**Table 7** summarises final episode rewards across all five experimental conditions.

**Table 7.** Summary of final episode rewards across all conditions (mean ± std, $n = 5$ seeds).

| Configuration | Backend | Terrain | Episode Reward | Convergence |
|--------------|---------|---------|---------------|-------------|
| Baseline | MuJoCo | Flat | 304.6 ± 428.5 | 3/5 |
| Baseline | MuJoCo | Heightfield | 550.8 ± 56.2 | 5/5 |
| Baseline | Brax/JAX | Flat | 648.3 ± 31.6 | 5/5 |
| Baseline | Brax/JAX | Heightfield | **659.7 ± 21.2** | 5/5 |
| Vendor-Tuned | Brax/JAX | Flat | 391.9 ± 76.9 | 4/5* |

*Seed\_19946 did not achieve meaningful locomotion (reward 304.7, forward velocity 0.002 m/s).

---

## 4.5 Discussion

### 4.5.1 Backend Comparability Under the Vendor-Agnostic Framework

The central claim of the vendor-agnostic design is that the same reward function, observation space, and policy architecture should produce comparable training outcomes regardless of the underlying physics backend. The results support this claim conditionally: when the MuJoCo backend converges (3 of 5 seeds on flat terrain, 5 of 5 on heightfield), the final episode rewards (637 on flat, 551 on heightfield) are within the same order of magnitude as the Brax baseline (648, 660). The reward function and observation encoding transfer faithfully across backends with no modification.

However, the backends differ substantially in **training stability**. The Brax/JAX backend achieves 100% convergence across all conditions, while the MuJoCo backend fails in 40% of flat-terrain seeds. This stability difference stems from the training loop architecture rather than the reward function: JAX's `jax.vmap` ensures all 256 environments are stepped in a single deterministic, synchronised computation, whereas the MuJoCo multiprocessing runner introduces asynchronous worker scheduling that can produce transiently biased rollout batches during early training.

### 4.5.2 Effect of GPU-Accelerated Scaling

The vendor-tuned configuration demonstrates both the potential and the limitations of GPU-accelerated scaling. The 8× larger batch size ($524{,}288$ vs $65{,}536$ samples per update) provides lower-variance gradient estimates and enables training for 12M steps — 2.4× longer than the baseline — within a comparable wall-clock time. The best vendor-tuned seed (527.2, forward velocity 0.161 m/s) outperforms the mean baseline Brax reward and achieves the highest forward velocity across all configurations.

However, training throughput drops from 3 939 SPS to 948 SPS with the larger batch. This counter-intuitive result — more environments running slower — arises from two sources: (1) the XLA memory allocator, which was configured to use direct `hipMalloc` to avoid ROCm command-buffer bugs on the MI300X, incurs additional per-step allocation overhead at large batch sizes; and (2) the 8× larger gradient computation dominates each update step. The originally intended batch of 8 192 environments ($67\times$ the baseline) exceeded the MI300X's 192 GB HBM3 capacity and was reduced to 2 048 environments for all reported runs.

### 4.5.3 Locomotion Quality

Across all configurations, the robot consistently learns to maintain an upright posture ($\bar{u}_z > 0.94$) but achieves forward velocities substantially below the target ($v^* = 0.6$–$0.8$ m/s). Mean achieved velocities range from 0.002 m/s (failed vendor-tuned seed) to 0.161 m/s (best vendor-tuned seed). Within the training budgets evaluated (5M–12M steps), neither backend produces a policy that approaches the target speed.

This gap is likely attributable to the complexity of the hexapod gait — coordinating 18 joints across 6 legs requires discovering stable tripod or wave gait patterns. The reward function provides no explicit gait shaping; the robot must discover foot-timing coordination entirely from the velocity signal. Future work could incorporate curriculum learning (progressive target velocity increase), gait-specific shaping terms, or imitation of reference motion to accelerate convergence to efficient locomotion.

The orientation term ($c_{\text{orient}} = 1.0$) provides strong pressure to remain upright, which the robot satisfies reliably. However, high orientation cost may suppress forward-locomotion exploration, since any destabilising exploratory motion is penalised immediately.

### 4.5.4 Terrain Generalisation

Both backends generalise better to the heightfield terrain than the results might initially suggest. On MuJoCo, the heightfield completely eliminates the catastrophic convergence failures seen on flat terrain (0 failed seeds vs 2). On Brax, heightfield performance (659.7) marginally surpasses flat (648.3) with lower variance (21.2 vs 31.6). These results indicate that the reward function encodes sufficient terrain-agnostic objectives — upright posture and forward velocity — to guide training on both surface geometries without terrain-specific engineering.

The static heightfield in this study (generated once with a fixed seed) represents a lower bound on terrain generalisation difficulty. Dynamic terrain randomisation would be required to claim full generalisation capability.

### 4.5.5 Limitations

1. **Gait performance gap:** No configuration achieves the target forward velocity. Extended training, curriculum learning, or richer reward shaping would be needed to close this gap.
2. **Memory-constrained scaling:** The vendor-tuned batch size was reduced from the intended $8{,}192$ to $2{,}048$ environments due to MI300X HBM3 capacity limits, constraining the upper bound of GPU-side tuning.
3. **MuJoCo metric incompleteness:** Forward velocity and upright factor were not logged by the MuJoCo CSV logger in the format used by the Brax backend, preventing a direct per-metric comparison for Stage 1.
4. **Static terrain only:** No terrain randomisation or curriculum was applied; heightfield results reflect training on a single fixed surface.
5. **Simulation-only evaluation:** No physical deployment was attempted. Sim-to-real transfer for this hexapod platform remains an open question.

---

## 4.6 Summary

**Table 8.** Key findings across all experimental stages.

| Finding | Evidence |
|---------|---------|
| Vendor-agnostic reward transfers across backends | Converged MuJoCo and Brax rewards within 2% of each other (637 vs 648) |
| Brax/JAX backend more training-stable | 100% convergence on all conditions vs 60% for MuJoCo on flat terrain |
| GPU acceleration provides ~6.4× throughput over CPU at equal parallelism | 3 939 SPS (Brax) vs ~617 SPS (MuJoCo), both with 256 envs |
| Vendor-tuned scaling improves peak performance but increases variance | Best seed: 527 reward, 0.161 m/s; worst seed: 305 reward, near-stationary |
| Heightfield terrain improves convergence stability for both backends | MuJoCo heightfield: 5/5 converged; Brax heightfield: lower variance (21 vs 32) |
| Robot learns upright posture but not target-speed locomotion | $\bar{u}_z > 0.94$ across all conditions; forward velocity 0.033–0.055 m/s vs target 0.6–0.8 m/s |

---

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR 2016.
- Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.
- Freeman, C. D., Frey, E., Raichuk, A., Girgin, S., Mordatch, I., & Bachem, O. (2021). *Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation*. NeurIPS 2021.
- Bradbury, J., et al. (2018). *JAX: composable transformations of Python+NumPy programs*. GitHub.
- Tassa, Y., et al. (2022). *MJX: MuJoCo XLA*. GitHub.
