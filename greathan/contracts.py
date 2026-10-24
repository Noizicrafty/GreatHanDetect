from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import math

import numpy as np


@dataclass
class Person:
    box: list[float]
    confidence: float
    keypoints: Any = None
    keypoint_confidence: Any = None
    track_id: int = 0


@dataclass
class Observation:
    persons: list[Person] = field(default_factory=list)
    objects: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_tensors(tensors: dict, specs: dict, label: str) -> None:
    if not isinstance(tensors, dict) or set(tensors) != set(specs):
        raise ValueError(f"{label}: tensor names must be {list(specs)}")
    for name, spec in specs.items():
        value = tensors[name]
        shape = tuple(spec["shape"])
        dtype = np.dtype(spec.get("dtype", "float32"))
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{label}.{name}: expected {shape} {dtype}; got "
                             f"{getattr(value, 'shape', None)} {getattr(value, 'dtype', None)}")
        if value.nbytes > 64 * 1024 * 1024 or not np.isfinite(value).all():
            raise ValueError(f"{label}.{name}: tensor is too large or contains non-finite numbers")


def finite_number(value, name: str, minimum=0.0, maximum=1e9) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name}: expected a finite number in [{minimum}, {maximum}]")
    return number
