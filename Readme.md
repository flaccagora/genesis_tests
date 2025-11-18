# CV tests for Template based mesh reconstruction

The project now follows a standard `src/` layout so that every Python module can
be imported directly once the directory is on `PYTHONPATH`. Top-level assets
(`assets/`, `datasets/`, `config/`, `lightning_logs/`, etc.) stay where they are
so existing paths keep working.

## Project Layout

- `src/`: Python source tree
  - `train/`: Lightning module, data module, and training entry-point
  - `data/`: dataset definitions and dataloaders (`datasets.py`, etc.)
  - `models/`: network definitions plus standalone training helpers
  - `evaluation/`: offline eval/visualisation scripts
  - `tests/`: smoke tests and manual experiments
  - `simul/`: simulation setups, routines, and data-generation tools
  - `utils/`: shared helpers such as `configurator.py` and rotation helpers
- `config/`: lightweight override files consumed via `utils.configurator`
- `assets/`, `datasets/`, `lightning_logs/`: data, assets, and log outputs

Always run Python commands from the project root and ensure `src/` is discoverable:

```bash
export PYTHONPATH=src
```

You can also prefix one-off commands with `PYTHONPATH=src ...`.

## Usage

- **Lightning training (recommended):**

  ```bash
  PYTHONPATH=src python -m train.train config/lightning.py \
      --batch_size=64 --train_dir=datasets/data_Torus_5
  ```

  This loads defaults from `train/train.py`, applies overrides from the
  provided config file, and then processes any `--key=value` flags.

- **Dataset generation:**

  ```bash
  PYTHONPATH=src python -m simul.generation config/data_generation.py
  ```

- **Standalone model training/testing:**

  ```bash
  PYTHONPATH=src python -m models.train config/train.py
  PYTHONPATH=src python -m evaluation.evaluate config/test.py
  PYTHONPATH=src python -m tests.scene
  ```

Refer to the files under `config/` for tunable parameters. All scripts rely on
`utils.configurator.apply_overrides`, so config files and CLI flags work
consistently across entry points.
