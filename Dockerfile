FROM rocm/jax:rocm7.2.3-jax0.8.2-py3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    git \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with ROCm support matching the base image (rocm/jax:rocm7.2.3).
# pytorch.org ships wheels per ROCm minor; update the tag here if a newer build drops.
RUN python -m pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/rocm7.2

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt


COPY . /workspace

RUN chmod +x /workspace/tools/run_pipeline.sh

CMD ["bash"]
