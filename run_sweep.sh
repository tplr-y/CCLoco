# run_sweep.sh
#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1

# 1) create the sweep and capture the ID that wandb prints
SWEEP_ID=$(wandb sweep "$CONFIG" 2>&1 | \
           awk '/wandb agent/{print $NF}')

echo "🪄  Created sweep → $SWEEP_ID"

# 2) launch the agent for that sweep
wandb agent "$SWEEP_ID"