# Scripts

train_nohup.sh — simple launcher to run multiple training jobs with nohup

How to use

1. Make the script executable (once):

   chmod +x scripts/train_nohup.sh

2. (Optional) Do a dry-run to preview commands:

   ./scripts/train_nohup.sh --dry-run

3. Launch jobs (optionally limit GPUs):

   CUDA_VISIBLE_DEVICES=0 ./scripts/train_nohup.sh

Logs and PIDs

- Logs are created under `./logs/` in the project root.
- For each launched job a `.pid` file with the process id is stored next to the log.

Customisation

Edit `scripts/train_nohup.sh` to change which configurations are run (see BATCH_SIZES, LRS,
CONFIG and DATA_DIR variables near the top of the file).

Notes

- The script assumes the project root uses `src/` as the Python package root. It calls
  training with `env PYTHONPATH=src python -m train.train config/lightning.py` and forwards
  simple `--key=value` overrides to the project's configurator.
- The script uses `nohup` so jobs survive the terminal session; check logs for progress.
