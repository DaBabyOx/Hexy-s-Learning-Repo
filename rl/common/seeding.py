from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_numpy(seed: int) -> None:
    np.random.seed(seed)


def seed_python(seed: int) -> None:
    random.seed(seed)


def seed_torch(seed: int) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_all(seed: int) -> None:
    seed_python(seed)
    seed_numpy(seed)
    seed_torch(seed)


def fold_in_seed(seed: int, offset: int) -> int:
    return int((seed + 9973 * offset) % (2**31 - 1))
