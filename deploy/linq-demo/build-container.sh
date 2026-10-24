#!/usr/bin/env bash
set -u
cd /opt/linq-demo
docker compose -f compose.linq-demo.yaml build --pull
status=$?
printf '%s\n' "$status" > build.exit
exit "$status"
