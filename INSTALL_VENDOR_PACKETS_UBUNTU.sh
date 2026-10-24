#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -ne 1 ]]; then echo "Usage: $0 /path/to/packets" >&2; exit 2; fi
PACKETS="$(realpath "$1")"
if [[ ! -f "$PACKETS/requirements.txt" ]]; then echo "Missing vendor requirements" >&2; exit 2; fi
if [[ ! -f .venv/bin/activate ]]; then echo "Create .venv first" >&2; exit 2; fi
source .venv/bin/activate
python - <<'PY'
import sys
if sys.version_info[:3] != (3,12,11):
    sys.exit('Vendor documentation requires Python 3.12.11')
PY
python -m pip install -r "$PACKETS/requirements.txt"
python tools/doctor.py
