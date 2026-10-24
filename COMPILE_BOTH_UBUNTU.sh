#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python tools/inspect_onnx.py --part detector
python tools/inspect_onnx.py --part pose
python tools/compile_tpu.py --part detector
python tools/compile_tpu.py --part pose
