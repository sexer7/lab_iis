#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1
