# Vendor-Agnostic RL Bootstrap

This repo now contains a very small first-pass bootstrap for loading the robot
meshes in both MuJoCo and Brax without any actuation or movement.

Key point: you do not need to convert the STL files to URDF just to get the
first scene loading. MuJoCo can reference STL meshes directly through MJCF, and
Brax can load that same MJCF. URDF can still be useful later if you want a more
portable robot-description format, but it is optional for this stage.

## Files

- `models/hexapod_static.xml`: static MJCF scene that references every STL in
  `STLFILES`
- `tools/generate_mjcf.py`: regenerates the MJCF from the mesh folder
- `scripts/load_mujoco.py`: smoke test for MuJoCo loading
- `scripts/load_brax.py`: smoke test for Brax loading

## Expected layout

The current scene assumes:

- `BODY.stl` is the trunk mesh
- `Coxa.*.*.stl`, `Femur.*.*.stl`, `Tibia.*.*.stl` are the six leg segments
- `F`, `C`, `B` mean front, center, back
- `L`, `R` mean left, right

## Usage

If you already have Python plus `mujoco` and `brax` installed:

```powershell
python tools/generate_mjcf.py
python scripts/load_mujoco.py
python scripts/load_brax.py
python scripts/test_mujoco_control.py
python scripts/view_mujoco.py
```

## Docker

If your host Python does not see the GPU reliably, use Docker instead.

Build and open the container:

```powershell
docker compose build
docker compose run --rm vendor-agnostic-rl
```

Inside the container:

```bash
python tools/generate_mjcf.py
python scripts/load_mujoco.py
python scripts/load_brax.py
```

Notes:

- `docker-compose.yml` defaults to `JAX_ACCELERATOR=cuda12`
- if you want CPU only in PowerShell:

```powershell
$env:JAX_ACCELERATOR="cpu"
docker compose build
```

- this setup is aimed at NVIDIA passthrough today because Docker GPU support is
  much smoother there; we can add a ROCm variant next if you want AMD support
- for NVIDIA GPUs, make sure Docker Desktop GPU support and the NVIDIA
  Container Toolkit path are working on the host first

## Next step

Once the meshes load, the next clean step is:

1. verify mesh orientation and scale
2. add real joint pivots for coxa/femur/tibia
3. add actuators and limits
4. optionally export a URDF if another toolchain needs it
