#!/usr/bin/env bash
# Run inside a clean Git clone. Requires Docker Engine + Compose >= 2.20.
set -euo pipefail
cd "$(dirname "$0")/../.."
git diff --quiet && git diff --cached --quiet || {
  echo 'Commit or stash local edits before deployment.' >&2; exit 1;
}
git pull --ff-only origin main
docker compose up -d --build --wait --wait-timeout 120
docker compose --profile check run --rm check
docker compose ps
