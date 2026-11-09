"""Minimum corrective action under SAF supply disruption (Eq. 11; Sections 2.8 and 4.4 of the paper).

Levers, by year t = 2025..2050 and band d:
  x^A_t      alternative bio-SAF (t), cap = alt_supply_cap_share * F_t, abates e*theta_bio per t
  x^R_{d,t}  early retirements (aircraft), cap = slots not used by the scenario itself;
             each saves f_{d,t} t fossil fuel per year for R years, abates e per t
  x^O_{d,t}  extra ATM fuel savings (t), cap = (corrective_max - nominal recovery) * pool_{d,t};
             abates e per t
Supply is binding under disruption, so fuel saved displaces fossil kerosene (Eq. 5).

Objective: minimum cost (EUR) subject to restoring cumulative 2025-2050 AND 2050 emissions to the
pathway's nominal values. Residuals u >= 0 carry a prohibitive penalty and measure the part of the
gap the levers cannot close within their caps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from . import system as sy
from .params import Params, crf


@dataclass
class LeverData:
    f_save: np.ndarray       # 4 x T fuel saved per early retirement per year (t)
    slots: np.ndarray        # 4 x T available slots (aircraft)
    cost_R: np.ndarray       # 4 x T net cost per early retirement (EUR, over R years)
    atm_cap: np.ndarray      # 4 x T extra ATM fuel savings available (t)
    cost_O: float            # net EUR per t fuel saved by ATM
    alt_cap: np.ndarray      # T alternative SAF cap (t)
    cost_A: float            # EUR per t alternative SAF (premium over fossil)


def aircraft_economics(P: Params) -> dict:
    """Per-class prices (EUR) used for the renewal lever."""
    price = P.arr("aircraft_classes.list_price_musd") * 1e6 * P["aircraft_classes.market_to_list_ratio"] \
        * P["aircraft_classes.usd_to_eur"]
    resid = P["aircraft_classes.residual_value_share_old"] * price
    return {"K_new": price, "V_old": resid}


def lever_data(P: Params, base: sy.Base, nom: sy.Result) -> LeverData:
    T = len(nom.years)
    R = int(P["corrective.advance_years"])
    cr = crf(P["corrective.discount_rate"], int(P["corrective.amortisation_years"]))
    eco = aircraft_economics(P)
    pf = P["saf.price_fossil_eur_t"]
    f_save = np.zeros((4, T)); slots = np.zeros((4, T)); cost_R = np.zeros((4, T)); atm_cap = np.zeros((4, T))
    rec_nom = 1 - nom.atm_ratio / base.atm_ratio0[:, None]          # nominal recovered fraction
    corr_max = P["atm.corrective_max_fraction"]
    for d in range(4):
        bf = base.fleets[d]; fp = nom.fleet_paths[d]
        fuel_per_km_new = base.fuel_eff0[d] / base.km0[d] / fp.m_mean[0]     # t/km at multiplier 1
        f_save[d] = bf.util_km * fuel_per_km_new * (fp.m_oldest - fp.m_new) * (1 - nom.pi[d])
        slots[d] = np.maximum(fp.slots - fp.early_retired, 0.0)
        K = bf.km_weights @ eco["K_new"]; V = bf.km_weights @ eco["V_old"]
        cost_R[d] = R * (cr * (K - V) - pf * f_save[d])
        pool = nom.fuel_eff[d] / (1 + nom.atm_ratio[d]) * base.atm_ratio0[d] * (1 - nom.pi[d])
        atm_cap[d] = np.maximum(corr_max - rec_nom[d], 0.0) * pool
    atm_cap[:, 0] = 0.0
    cost_O = P["atm.gross_cost_eur_per_t_co2"] * P.e_ttw - pf
    alt_cap = P["saf.alt_supply_cap_share"] * nom.fuel.sum(0)
    cost_A = P["saf.price_bio_eur_t"] - pf
    return LeverData(f_save, slots, cost_R, atm_cap, cost_O, alt_cap, cost_A)


def solve(P: Params, base: sy.Base, key: str, delta: float, r: float, r_ptl: float | None = None,
          levers: tuple = ("A", "R", "O"), objective: str = "cost",
          targets: tuple = ("cum", "2050")) -> dict:
    """Solve the corrective LP for pathway `key` under disruption (delta, r).

    targets: which nominal outcomes must be restored -- the cumulative 2025-2050 budget ("cum"),
    the 2050 level ("2050"), or both (default)."""
    lev = sy.SCENARIOS[key].levers
    nom = sy.run(P, base, lev)
    dis = sy.run(P, base, lev, saf_delay=delta, saf_r=r, saf_r_ptl=r_ptl, saf_fixed_fuel=nom.fuel.sum(0))
    gap = (dis.emissions - nom.emissions)[1:]                 # t CO2e, years 2025..2050
    T = len(gap)
    ld = lever_data(P, base, nom)
    e, R = P.e_wtw, int(P["corrective.advance_years"])
    thb = P["saf.theta_bio"]

    nA, nR, nO = T, 4 * T, 4 * T
    N = nA + nR + nO + 2
    M = np.zeros((T, nA + nR + nO))                           # abatement in year t from each decision
    for t in range(T):
        M[t, t] = e * thb
        for d in range(4):
            for tau in range(T):
                if tau <= t < tau + R:
                    M[t, nA + d * T + tau] = e * ld.f_save[d, tau + 1]
            M[t, nA + nR + d * T + t] = e
    A_ub = np.zeros((2, N)); A_ub[0, :-2] = -M.sum(0); A_ub[1, :-2] = -M[-1]
    A_ub[0, -2] = -1.0; A_ub[1, -1] = -1.0
    b_ub = np.array([-gap.sum(), -gap[-1]])
    keep = [i for i, tg in enumerate(("cum", "2050")) if tg in targets]
    A_ub, b_ub = A_ub[keep], b_ub[keep]

    bounds = [(0, ld.alt_cap[t + 1] if "A" in levers else 0) for t in range(T)]
    bounds += [(0, ld.slots[d, t + 1] if "R" in levers else 0) for d in range(4) for t in range(T)]
    bounds += [(0, ld.atm_cap[d, t + 1] if "O" in levers else 0) for d in range(4) for t in range(T)]
    bounds += [(0, None), (0, None)]

    pen = P["corrective.penalty_eur_per_t"]
    if objective == "cost":
        # Measures with negative net cost would be adopted in the nominal pathway unless barriers
        # block them; as corrective levers they are used only where needed. The ranking cost is
        # max(net cost, 0) plus a small charge per tonne CO2e abated (same unit for every lever),
        # with a tiny tie-break on the net cost so that cheaper negative-cost measures come first.
        # The reported cost uses the true net cost.
        true = np.concatenate([np.full(nA, ld.cost_A), ld.cost_R[:, 1:].ravel(), np.full(nO, ld.cost_O)])
        c = np.concatenate([np.maximum(true, 0.0) + 0.01 * M.sum(0) + 1e-6 * true, [pen, pen]])
    else:   # physical minimum: aircraft count / tonnes, residual penalised
        c = np.concatenate([np.full(nA, 1e-6), np.ones(nR), np.full(nO, 1e-6), [10.0, 10.0]])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    z = res.x
    xa = z[:nA]; xr = z[nA:nA + nR].reshape(4, T); xo = z[nA + nR:nA + nR + nO].reshape(4, T)
    ab = M * z[:-2]
    ab_cum = ab.sum(0)
    by = {"alt_saf": ab_cum[:nA].sum(),
          "renewal": ab_cum[nA:nA + nR].reshape(4, T).sum(1),
          "atm": ab_cum[nA + nR:].reshape(4, T).sum(1)}
    cost = ld.cost_A * xa.sum() + (ld.cost_R[:, 1:] * xr).sum() + ld.cost_O * xo.sum()
    return {"gap_cum": gap.sum(), "gap_2050": gap[-1], "alt_saf_t": xa.sum(), "aircraft": xr.sum(1),
            "atm_fuel_t": xo.sum(1), "abated_cum": by, "residual_cum": z[-2], "residual_2050": z[-1],
            "cost_eur": cost, "status": res.status}


def mac_table(P: Params, base: sy.Base, key: str = "S2", year: int = 2030) -> dict:
    """Static abatement cost (EUR per t CO2e, WTW) of each corrective lever, by band for renewal and ATM."""
    nom = sy.run(P, base, sy.SCENARIOS[key].levers)
    ld = lever_data(P, base, nom)
    t = int(np.nonzero(nom.years == year)[0][0])
    R = int(P["corrective.advance_years"])
    e = P.e_wtw
    renewal = ld.cost_R[:, t] / (R * e * ld.f_save[:, t])
    per_ac = e * ld.f_save[:, t]
    return {"renewal": renewal, "renewal_abatement_per_aircraft_t": per_ac,
            "atm": ld.cost_O / e,
            "alt_bio": ld.cost_A / (e * P["saf.theta_bio"]),
            "alt_ptl": (P["saf.price_ptl_eur_t"] - P["saf.price_fossil_eur_t"]) / (e * P["saf.theta_ptl"])}
