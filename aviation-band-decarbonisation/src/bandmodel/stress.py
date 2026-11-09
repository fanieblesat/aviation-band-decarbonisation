"""SAF supply-risk stress test (Sections 2.7 and 4.3 of the paper).

Disruption family: S_t(delta, r) = r * sigma^min_{t-delta} * F_t, applied to bio and/or synthetic
SAF. Targets are the pathway's own nominal outcomes (cumulative 2025-2050 budget and 2050 level)
plus an optional slack, or an external level shared by all pathways.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import system as sy
from .params import Params


def excess_grid(P: Params, base: sy.Base, key: str, deltas: np.ndarray, rs: np.ndarray):
    """Percentage excess over nominal of cumulative and 2050 emissions, on a (r, delta) grid."""
    lev = sy.SCENARIOS[key].levers
    nom = sy.run(P, base, lev)
    E0 = nom.emissions
    cum = np.zeros((len(rs), len(deltas))); end = np.zeros_like(cum)
    for i, r in enumerate(rs):
        for j, dl in enumerate(deltas):
            E = sy.run(P, base, lev, saf_delay=dl, saf_r=r, saf_fixed_fuel=nom.fuel.sum(0)).emissions
            cum[i, j] = 100 * (E[1:].sum() / E0[1:].sum() - 1)
            end[i, j] = 100 * (E[-1] / E0[-1] - 1)
    return cum, end


def r_star_closed_form(P: Params, base: sy.Base, key: str, slack: float = 0.0,
                       external_level: float | None = None) -> float:
    """Critical realisation rate for the 2050 level (Eq. 10), no delay.

    E(r) = e [F - r A] + e kappa R,  A = theta_bio S_bio + theta_ptl S_ptl (nominal)
    target: E(r) <= (1+slack) E_nom, or E(r) <= external_level."""
    nom = sy.run(P, base, sy.SCENARIOS[key].levers)
    F = nom.fuel[:, -1].sum()
    A = nom.theta_bio * nom.s_bio[-1] + nom.theta_ptl * nom.s_ptl[-1]
    Z = nom.kappa[-1] * nom.zep_equiv[:, -1].sum()
    target = (1 + slack) * nom.emissions[-1] if external_level is None else external_level
    return (F + Z - target / nom.e) / A


def r_star_numeric(P: Params, base: sy.Base, key: str, slack: float = 0.0,
                   external_level: float | None = None) -> float:
    lev = sy.SCENARIOS[key].levers
    nom = sy.run(P, base, lev)
    target = (1 + slack) * nom.emissions[-1] if external_level is None else external_level

    def f(r: float) -> float:
        return sy.run(P, base, lev, saf_r=r, saf_fixed_fuel=nom.fuel.sum(0)).emissions[-1] - target

    if f(1.0) > 0:
        return float("nan")
    if f(0.0) <= 0:
        return 0.0
    return brentq(f, 0.0, 1.0, xtol=1e-6)


def delay_sensitivity(P: Params, base: sy.Base, key: str, delay: float = 1.0) -> tuple[float, float]:
    """Percentage change in 2050 and cumulative emissions from a pure delay."""
    lev = sy.SCENARIOS[key].levers
    nom = sy.run(P, base, lev)
    E = sy.run(P, base, lev, saf_delay=delay, saf_fixed_fuel=nom.fuel.sum(0)).emissions
    return 100 * (E[-1] / nom.emissions[-1] - 1), 100 * (E[1:].sum() / nom.emissions[1:].sum() - 1)
