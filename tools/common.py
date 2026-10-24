"""Shared helpers. Explicitly implements YOLO letterbox preprocessing, without ImageNet normalization."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project_cfg():
    import yaml
    return yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))


def dest(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def letterbox(frame, size: int):
    import cv2
    h, w = frame.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw) / 2, (size - nh) / 2
    left, top = int(round(dx - 0.1)), int(round(dy - 0.1))
    right, bottom = size - nw - left, size - nh - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None]), (gain, float(left), float(top), w, h)


def unletterbox_xyxy(boxes, meta):
    gain, dx, dy, w, h = meta
    result = np.asarray(boxes, dtype=np.float32).copy()
    result[:, [0, 2]] = (result[:, [0, 2]] - dx) / gain
    result[:, [1, 3]] = (result[:, [1, 3]] - dy) / gain
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, w)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, h)
    return result


def ensure_clyv_path():
    p = ROOT / "vendor" / "smoking-detection"
    if not (p / "smokedet" / "fusion.py").exists():
        sys.exit("Original source absent: run 'python tools/bootstrap_source.py' first (Git required).")
    sys.path.insert(0, str(p))
    return p
