"""
Poor Man's Configurator.

Example usage from a training script:

    from utils.configurator import apply_overrides
    apply_overrides(globals())

This will first apply any config/*.py files that are passed on the command line
and then interpret `--key=value` style overrides.
"""

from __future__ import annotations

import sys
from ast import literal_eval
from typing import MutableMapping, Sequence

__all__ = ["apply_overrides"]


def _load_config_file(config_file: str, namespace: MutableMapping[str, object]) -> None:
    print(f"Overriding config with {config_file}:")
    with open(config_file) as f:
        file_contents = f.read()
        print(file_contents)
    exec(compile(file_contents, config_file, "exec"), namespace)


def _apply_kv_override(key: str, value: str, namespace: MutableMapping[str, object]) -> None:
    if key not in namespace:
        raise ValueError(f"Unknown config key: {key}")
    current_value = namespace[key]
    try:
        attempted_value = literal_eval(value)
    except (SyntaxError, ValueError):
        attempted_value = value
    if type(attempted_value) is not type(current_value):
        raise TypeError(
            f"Type mismatch for {key}: expected {type(current_value).__name__}, got {type(attempted_value).__name__}"
        )
    print(f"Overriding: {key} = {attempted_value}")
    namespace[key] = attempted_value


def apply_overrides(
    namespace: MutableMapping[str, object],
    argv: Sequence[str] | None = None,
) -> None:
    """
    Apply configuration overrides directly into the provided namespace.

    Parameters
    ----------
    namespace:
        Typically `globals()` from the calling module.
    argv:
        Explicit command-line style arguments. Defaults to `sys.argv[1:]`.
    """

    args = list(argv if argv is not None else sys.argv[1:])
    for arg in args:
        if "=" not in arg:
            if arg.startswith("--"):
                raise ValueError(f"Expected config file path, got flag: {arg}")
            _load_config_file(arg, namespace)
        else:
            if not arg.startswith("--"):
                raise ValueError(f"Expected --key=value format, got: {arg}")
            key, value = arg.split("=", 1)
            key = key[2:]
            _apply_kv_override(key, value, namespace)


if __name__ == "__main__":
    apply_overrides(globals())