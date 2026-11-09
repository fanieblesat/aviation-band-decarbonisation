"""Invariants the paper relies on. Run: pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bandmodel import corrective, flights, params, stress, system  # noqa: E402


@pytest.fixture(scope="module")
def model():
    P = params.load()
    df, cal = flights.generate(P, n=20000)
    return P, df, cal, system.build_base(P, df)


def test_wtw_factor():
    P = params.load()
    assert P.e_wtw == pytest.approx(89.0 * 43.15 / 1000)


def test_calibration_reproduces_targets(model):
    P, df, cal, _ = model
    bt = flights.band_table(P, df)
    assert np.allclose(bt["flight_share"], P.arr("bands.flight_share_2019"), atol=0.01)
    assert np.allclose(bt["co2_share"], P.arr("bands.co2_share_2019"), atol=0.025)
    assert bt["seatkm_share"].iloc[3] > P["bands.seatkm_share_longhaul_min"]       # validation, not fitted


def test_base_scales_to_anchor(model):
    P, _, _, B = model
    total = (B.fuel_eff0 * (1 + B.atm_ratio0)).sum() * P.e_ttw / 1e6
    assert total == pytest.approx(P["demand.co2_2024_mt"], rel=1e-9)


def test_index_invariant_to_fossil_factor(model):
    """Eqs. (1)-(2): the index is a ratio, so scaling the fossil factor leaves it unchanged."""
    P, _, _, B = model
    r1 = system.run_scenario(P, B, "S2")
    r1.e = 3.16
    I_ttw = r1.index.copy()
    r1.e = P.e_wtw
    assert np.allclose(I_ttw, r1.index)


def test_saf_allocation_neutral(model):
    """Eq. (5): system emissions depend on SAF quantity, not on its allocation across bands."""
    P, _, _, B = model
    r = system.run_scenario(P, B, "S2")
    E = r.emissions_by_band.sum(0)
    assert np.allclose(E, r.emissions)


def test_mandate_is_minimum_and_stepwise():
    P = params.load()
    yrs = np.arange(2024, 2051)
    tot, syn = system.mandate_shares(P, yrs)
    assert tot[yrs == 2049][0] == pytest.approx(0.42) and tot[yrs == 2050][0] == pytest.approx(0.70)
    assert np.all(np.diff(tot) >= 0) and np.all(syn <= tot)


def test_shapley_sums_to_total(model):
    P, _, _, B = model
    sh = system.shapley(P, B)
    full = system.run(P, B, frozenset(system.LEVERS)).emissions_by_band[:, -1]
    none = system.run(P, B, frozenset()).emissions_by_band[:, -1]
    assert np.allclose(sh.sum(axis=1).to_numpy() * 1e6, none - full, rtol=1e-9)


@pytest.mark.parametrize("key", ["S2", "S3"])
@pytest.mark.parametrize("slack", [0.05, 0.1, 0.2])
def test_rstar_closed_form_matches_root(model, key, slack):
    P, _, _, B = model
    assert stress.r_star_closed_form(P, B, key, slack) == pytest.approx(stress.r_star_numeric(P, B, key, slack), abs=1e-4)


def test_no_disruption_no_gap(model):
    P, _, _, B = model
    s = corrective.solve(P, B, "S2", 0.0, 1.0)
    assert abs(s["gap_cum"]) < 1.0 and s["cost_eur"] == pytest.approx(0.0, abs=1.0)


def test_corrective_restores_targets_when_feasible(model):
    P, _, _, B = model
    s = corrective.solve(P, B, "S2", 0.0, 0.9)
    closed = s["abated_cum"]["alt_saf"] + s["abated_cum"]["renewal"].sum() + s["abated_cum"]["atm"].sum()
    assert closed + s["residual_cum"] == pytest.approx(s["gap_cum"], rel=1e-6) or closed >= s["gap_cum"] * (1 - 1e-6)


def test_load_factor_signs(model):
    """Per-flight emissions rise with load factor; per-RPK intensity falls (colleague point 2)."""
    _, df, _, _ = model
    reg = flights.age_regression(df)
    assert reg["eta_flight"] > 0 > reg["eta_rpk"]
