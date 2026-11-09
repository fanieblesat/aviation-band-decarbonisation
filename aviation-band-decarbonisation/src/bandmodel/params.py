"""Load and expose model parameters from config/parameters.yaml.

Every parameter in the YAML carries value, status and source. `P` returns plain
values; `Params.table()` returns the provenance table used in the paper's
appendix, so no number reaches the manuscript without a recorded source.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "parameters.yaml"


def _unwrap(node: Any) -> Any:
    """Turn {'value': x, 'status': ..., 'source': ...} leaves into x, recursively."""
    if isinstance(node, dict):
        if "value" in node and ("status" in node or "source" in node):
            return node["value"]
        return {k: _unwrap(v) for k, v in node.items()}
    return node


@dataclass
class Params:
    raw: dict
    values: dict = field(init=False)

    def __post_init__(self) -> None:
        self.values = _unwrap(self.raw)

    # ---------------------------------------------------------------- access
    def __getitem__(self, path: str) -> Any:
        node = self.values
        for key in path.split("."):
            node = node[key]
        return node

    def arr(self, path: str) -> np.ndarray:
        return np.asarray(self[path], dtype=float)

    def with_overrides(self, overrides: dict[str, Any]) -> "Params":
        """Return a copy with dotted-path overrides applied to `value` fields."""
        raw = copy.deepcopy(self.raw)
        for path, val in overrides.items():
            node = raw
            keys = path.split(".")
            for k in keys[:-1]:
                node = node[k]
            leaf = node[keys[-1]]
            if isinstance(leaf, dict) and "value" in leaf:
                leaf["value"] = val
            else:
                node[keys[-1]] = val
        return Params(raw)

    # ------------------------------------------------------------ derived
    @property
    def e_wtw(self) -> float:
        """Well-to-wake factor of fossil jet fuel, t CO2e per t fuel."""
        return self["emission_factors.wtw_fossil_g_per_mj"] * self["emission_factors.lhv_mj_per_kg"] / 1000.0

    @property
    def e_ttw(self) -> float:
        return self["emission_factors.ttw_co2_per_t_fuel"]

    @property
    def years(self) -> np.ndarray:
        return np.arange(self["meta.base_year"], self["meta.horizon"] + 1)

    def table(self) -> list[dict]:
        """Flat provenance table: path, value, status, source."""
        rows: list[dict] = []

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict) and "value" in node and ("status" in node or "source" in node):
                rows.append({"path": prefix, "value": node["value"], "unit": node.get("unit", ""),
                             "status": node.get("status", ""), "source": node.get("source", "")})
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else k)

        walk(self.raw, "")
        return rows


def load(path: str | Path = DEFAULT_CONFIG) -> Params:
    with open(path, encoding="utf-8") as fh:
        return Params(yaml.safe_load(fh))


def step_path(steps: dict, years: np.ndarray, before: float = 0.0) -> np.ndarray:
    """Step function: value of the latest key <= year (regulation phrased per period)."""
    keys = np.array(sorted(int(k) for k in steps))
    vals = np.array([steps[k] if k in steps else steps[str(k)] for k in keys], dtype=float)
    out = np.full(len(years), before, dtype=float)
    for i, y in enumerate(years):
        idx = np.searchsorted(keys, y, side="right") - 1
        if idx >= 0:
            out[i] = vals[idx]
    return out


def linear_path(points: dict, years: np.ndarray) -> np.ndarray:
    keys = np.array(sorted(int(k) for k in points), dtype=float)
    vals = np.array([points[int(k)] if int(k) in points else points[str(int(k))] for k in keys], dtype=float)
    return np.interp(years, keys, vals)


def crf(rate: float, years: int) -> float:
    """Capital recovery factor."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
