#!/usr/bin/env bash
# Baseline A end-to-end entry point (Q1-Q3; Q4 deferred).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== P0 audit =="
python3 -m src.audit.audit_data

echo "== Q1-A =="
python3 -m src.baselines.q1a.run --config configs/baselines/Q1-A.yaml

echo "== Q2-A =="
python3 -m src.baselines.q2a.run --config configs/baselines/Q2-A.yaml

echo "== Q3-A =="
python3 -m src.baselines.q3a.run --config configs/baselines/Q3-A.yaml

echo "Done. See artifacts/baselines/Q{1,2,3}-A and reports/."
