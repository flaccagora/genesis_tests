#!/usr/bin/env bash
# Simple launcher to run multiple training jobs with nohup.
# By default jobs are started in the background (parallel). Pass --sequential to
# make the script wait for each job to finish before starting the next.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Default python module to run (uses src/ discovery via PYTHONPATH)
ENTRY_MODULE="train.train"

# Example parameter grids. Edit as needed.
BATCH_SIZES=(32 64)
LRS=(0.001 0.0001)
CONFIG="config/lightning.py"
DATA_DIR="datasets/data_Torus_5"

# GPU control: set CUDA_VISIBLE_DEVICES before calling this script to control visibility.

usage() {
  cat <<-USAGE
Usage: $(basename "$0") [--dry-run] [--sequential]

This script launches a small grid of training jobs using nohup.
By default jobs are started in the background (parallel). Use --sequential to wait
for each job to finish before starting the next.

Environment:
  Export PYTHONPATH=src is handled by the script when launching commands.

Options:
  --dry-run     Print the commands that would be executed without launching them.
  --sequential  Run jobs one at a time: wait for each job to finish before launching the next.

To limit GPUs, prefix invocation with e.g.:
  CUDA_VISIBLE_DEVICES=0 $(basename "$0") --sequential

Edit the BATCH_SIZES, LRS, CONFIG and DATA_DIR variables near the top of this file to
customise the grid of experiments.
USAGE
}

DRY_RUN=0
SEQUENTIAL=0
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --sequential)
      SEQUENTIAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

run_idx=0
for B in "${BATCH_SIZES[@]}"; do
  for LR in "${LRS[@]}"; do
    run_idx=$((run_idx+1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="train_${TIMESTAMP}_b${B}_lr${LR}.log"
    PIDFILE="$LOG_DIR/${LOGFILE}.pid"

    # Build overrides string (these depend on the project's configurator -- flags of the
    # form --key=value are forwarded and applied by utils.configurator)
    OVERRIDES=("--batch_size=${B}" "--learning_rate=${LR}" "--train_dir=${DATA_DIR}")

    CMD=(env PYTHONPATH=src python -m "$ENTRY_MODULE" "$CONFIG")
    for o in "${OVERRIDES[@]}"; do
      CMD+=("${o}")
    done

    # Join for printing
    CMD_STR="${CMD[*]}"

    if [[ $DRY_RUN -eq 1 ]]; then
      if [[ $SEQUENTIAL -eq 1 ]]; then
        echo "[DRY] (sequential) nohup ${CMD_STR} > $LOG_DIR/$LOGFILE 2>&1 & echo \$! > $PIDFILE; wait \$(cat $PIDFILE)"
      else
        echo "[DRY] nohup ${CMD_STR} > $LOG_DIR/$LOGFILE 2>&1 & echo \$! > $PIDFILE"
      fi
    else
      echo "Launching job #${run_idx}: batch=${B}, lr=${LR} -> $LOGFILE"
      nohup ${CMD_STR} > "$LOG_DIR/$LOGFILE" 2>&1 &
      pid=$!
      echo $pid > "$PIDFILE"
      echo "  PID=$pid  log=$LOG_DIR/$LOGFILE"

      if [[ $SEQUENTIAL -eq 1 ]]; then
        # Wait for the launched job to finish. If it fails, stop launching further jobs.
        echo "  Waiting for PID $pid to finish..."
        wait $pid
        rc=$?
        echo "  PID $pid exited with code $rc"
        if [[ $rc -ne 0 ]]; then
          echo "Job #${run_idx} failed (exit code $rc). Stopping further jobs."
          exit $rc
        fi
      else
        # brief sleep so timestamps differ slightly and to avoid hammering scheduler
        sleep 0.5
      fi
    fi
  done
done

if [[ $SEQUENTIAL -eq 1 ]]; then
  echo "Completed $run_idx sequential jobs. Logs: $LOG_DIR"
else
  echo "Launched $run_idx jobs. Logs: $LOG_DIR"
fi
