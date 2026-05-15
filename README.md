# Hexy's Learning Codes Repository
This repo contains a vendor-agnostic PPO pipeline for a custom hexapod MJCF.

## Quick Start
1) Create or activate the Python environment.
2) Install dependencies from requirements.txt.
3) Run a baseline PPO sweep for MuJoCo or Brax.

### Baseline (identical hyperparams)
MuJoCo (PyTorch):
```
python -m rl.cli --config configs/ppo_baseline.json --backend mujoco --terrain flat
```

Brax (MJX / JAX):
```
python -m rl.cli --config configs/ppo_baseline.json --backend brax --terrain flat
```

### Vendor-tuned runs
```
python -m rl.cli --config configs/ppo_vendor_tuned.json --backend brax --terrain flat
```

### Uneven terrain runs
```
python -m rl.cli --config configs/ppo_baseline.json --backend mujoco --terrain heightfield
python -m rl.cli --config configs/ppo_baseline.json --backend brax --terrain heightfield
```

## Docker
Build the image:
```
docker compose build
```

Run the full pipeline (baseline flat -> tuned -> baseline uneven):
```
docker compose run --rm vendor-agnostic-rl-train
```

ROCm (MI300X) notes:
- The Docker image uses the ROCm JAX base: `rocm/jax:rocm7.2.3-jax0.8.2-py3.11`.
- Switch to the py3.12 variant by editing the `FROM` line in Dockerfile.

To override which backends run in each stage:
```
BACKENDS_BASELINE="mujoco brax" BACKENDS_UNEVEN="mujoco brax" TUNED_BACKEND=brax \
	docker compose run --rm vendor-agnostic-rl-train
```

## Outputs
Each run writes to runs/<experiment>/<backend>/<terrain>/seed_<seed>:
- metrics.csv: scalar metrics over steps and wall time
- checkpoints/: framework checkpoint + unified policy export (npz)

## Notes
- The reward is identical across backends: forward velocity tracking with control, orientation, and joint-limit penalties.
- Unified policy checkpoints allow loading in either backend with the same MLP weights.
