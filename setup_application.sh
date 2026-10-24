#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements/application.txt
echo 'Optional cloud codec: install organiser tpu_pycloud_format wheel into .venv.'
echo 'Start: bash start_application.sh'
