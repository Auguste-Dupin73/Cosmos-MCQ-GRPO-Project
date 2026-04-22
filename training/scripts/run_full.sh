#!/usr/bin/env bash
set -euo pipefail

python training/train_grpo.py --config training/configs/full.yaml "$@"
