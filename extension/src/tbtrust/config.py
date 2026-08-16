"""Tiny YAML config system.

Layering: base `configs/default.yaml`, then an experiment yaml, then optional
`key.subkey=value` overrides from the command line. Kept intentionally minimal
(no Hydra) so a new team member can read it in two minutes and experiments are
just committed yaml files -- which keeps people out of each other's code.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, other: dict) -> dict:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(value: str) -> Any:
    """Turn a CLI string into an int/float/bool/list/None when it looks like one."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_override(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(*paths: str | Path, overrides: list[str] | None = None) -> dict:
    """Load and merge one or more yaml files, then apply `k.k=v` overrides."""
    cfg: dict = {}
    for p in paths:
        with open(p) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override '{ov}' must be key=value")
        key, val = ov.split("=", 1)
        apply_override(cfg, key.strip(), _coerce(val.strip()))
    return cfg


def load_experiment(experiment_yaml: str | Path, config_dir: str | Path = "configs",
                    overrides: list[str] | None = None) -> dict:
    """Load default.yaml + an experiment file from config_dir.

    `experiment_yaml` may be either a path that already resolves (the documented
    CLI usage, `--config configs/loco_montgomery.yaml`) or a bare filename inside
    `config_dir`. Previously only the first worked: default.yaml was joined to
    config_dir but the experiment file was not, so the bare-name form this
    docstring describes raised FileNotFoundError.
    """
    config_dir = Path(config_dir)
    experiment_path = Path(experiment_yaml)
    if not experiment_path.exists():
        candidate = config_dir / experiment_path
        if not candidate.exists():
            raise FileNotFoundError(
                f"config '{experiment_yaml}' not found, and neither is "
                f"'{candidate}'. Pass a path that exists, or a filename inside "
                f"--config-dir ({config_dir})."
            )
        experiment_path = candidate
    return load_config(config_dir / "default.yaml", experiment_path, overrides=overrides)
