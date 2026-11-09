"""Band-level system model: scenarios, SAF accounting and well-to-wake emissions.

For band d and year t (Eqs. 1-2 of the paper, well-to-wake):

  F_{d,t} = F^eff_{d,0} * (A_{d,t}/A_{d,0}) * Phi_{d,t} * (1 + a_{d,t}) * (1 - pi_{d,t})     fuel burned
  R_{d,t} = F^eff_{d,0} * (A_{d,t}/A_{d,0}) * Phi_{d,t} * (1 + a_{d,t}) * pi_{d,t}           fuel-equivalent of ZEP flights
  E_t     = e * ( sum_d F_{d,t} - theta_bio S^bio_t - theta_ptl S^ptl_t ) + e * kappa_t * sum_d R_{d,t}

SAF enters as quantities (bio and synthetic) because ReFuelEU sets a minimum
fleet-wide share and supply bounds it from above (Eq. 4); for a given quantity
the allocation across bands does not change E_t (Eq. 5). Per-band emissions for
reporting allocate SAF pro rata to fuel.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import factorial

import numpy as np
import pandas as pd

from . import fleet as fl
from .params import Params, linear_path, step_path

LEVERS = ("renewal", "atm", "saf", "zep")


@dataclass
class Base:
    fuel_eff0: np.ndarray        # base-year ATM-efficient fuel by band (t)
    atm_ratio0: np.ndarray       # excess / efficient fuel by band
    km0: np.ndarray              # base-year km by band (aircraft-km, scaled to the CO2 anchor)
    fleets: list
    flights0: np.ndarray         # base-year flights by band (scaled)


def build_base(P: Params, df: pd.DataFrame) -> Base:
    """Scale the flight sample to the 2024 CO2 anchor and extract band quantities."""
    total_fuel = P["demand.co2_2024_mt"] * 1e6 / P.e_ttw
    k = total_fuel / df["fuel_t"].sum()
    g = df.groupby("band")
    fuel = g["fuel_t"].sum().to_numpy() * k
    exc = g["fuel_atm_excess_t"].sum().to_numpy() * k
    eff = fuel - exc
    km_bc = np.zeros((4, 3))
    for (b, c), v in df.groupby(["band", "aclass"])["distance_km"].sum().items():
        km_bc[b, c] = v * k
    fleets = fl.band_fleets(P, km_bc)
    return Base(eff, exc / eff, km_bc.sum(axis=1), fleets, g.size().to_numpy() * k)


@dataclass
class Scenario:
    name: str
    levers: frozenset
    growth: float | None = None


SCENARIOS = {
    "S0": Scenario("Reference (no new measures)", frozenset()),
    "S1": Scenario("Operational efficiency", frozenset({"atm"})),
    "S2": Scenario("SAF mandate", frozenset({"atm", "saf"})),
    "S3": Scenario("Low-emission propulsion", frozenset({"atm", "saf", "zep", "renewal"})),
}


@dataclass
class Result:
    years: np.ndarray
    fuel: np.ndarray             # 4 x T fuel burned
    zep_equiv: np.ndarray        # 4 x T fuel-equivalent of ZEP flights
    fuel_eff: np.ndarray         # 4 x T ATM-efficient fuel incl. ZEP flights (before pi)
    atm_ratio: np.ndarray        # 4 x T
    phi: np.ndarray              # 4 x T fleet intensity index
    pi: np.ndarray               # 4 x T
    s_bio: np.ndarray            # T
    s_ptl: np.ndarray            # T
    kappa: np.ndarray            # T
    e: float
    theta_bio: float
    theta_ptl: float
    fleet_paths: list = field(default_factory=list)

    @property
    def emissions(self) -> np.ndarray:
        """System WTW emissions, t CO2e per year."""
        return self.e * (self.fuel.sum(0) - self.theta_bio * self.s_bio - self.theta_ptl * self.s_ptl) \
            + self.e * self.kappa * self.zep_equiv.sum(0)

    @property
    def emissions_by_band(self) -> np.ndarray:
        F = self.fuel.sum(0)
        abate = self.theta_bio * self.s_bio + self.theta_ptl * self.s_ptl
        share = np.divide(self.fuel, F, out=np.zeros_like(self.fuel), where=F > 0)
        return self.e * (self.fuel - share * abate) + self.e * self.kappa * self.zep_equiv

    @property
    def index(self) -> np.ndarray:
        E = self.emissions
        return 100 * E / E[0]


def mandate_shares(P: Params, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = P["saf.base_share_2024"]
    tot = np.maximum(step_path(P["saf.mandate_steps"], years, before=base), base)
    syn = step_path(P["saf.synthetic_steps"], years, before=0.0)
    return tot, syn


def run(P: Params, base: Base, levers: frozenset, saf_delay: float = 0.0, saf_r: float = 1.0,
        saf_r_ptl: float | None = None, growth: float | None = None,
        saf_fixed_fuel: np.ndarray | None = None) -> Result:
    """Simulate one lever combination. SAF disruption: share path delayed by `saf_delay` years
    and scaled by `saf_r` (bio) / `saf_r_ptl` (synthetic, defaults to saf_r).
    If `saf_fixed_fuel` is given, SAF quantities are computed on that fuel path (supply view),
    so that corrective fuel savings displace fossil kerosene only."""
    years = P.years
    T = len(years)
    t0 = years[0]
    g = P["demand.flights_growth_per_year"] if growth is None else growth
    act = (1 + g) ** (years - t0)

    phi = np.zeros((4, T)); paths = []
    for d in range(4):
        km = base.km0[d] * act
        accel = "renewal" in levers
        fp = fl.simulate(P, base.fleets[d], km, retire_age=20 if accel else None, accelerate=accel)
        phi[d] = fp.phi_index
        paths.append(fp)

    rec = np.zeros(T)
    if "atm" in levers:
        rec = P["atm.recoverable_fraction"] * np.clip((years - t0) / (P["atm.recovery_full_year"] - t0), 0, 1)
    atm = base.atm_ratio0[:, None] * (1 - rec)[None, :]

    pi = np.zeros((4, T)); kappa = np.zeros(T)
    if "zep" in levers:
        eis = P["zep.entry_year"]
        ramp = np.clip((years - eis + 1) / (2050 - eis + 1), 0, 1)
        pi = P.arr("zep.penetration_2050")[:, None] * ramp[None, :]
        kappa = linear_path(P["zep.upstream_factor"], years)

    gross = base.fuel_eff0[:, None] * act[None, :] * phi * (1 + atm)
    fuel = gross * (1 - pi)
    zep_eq = gross * pi

    if "saf" in levers:
        tot, syn = mandate_shares(P, years - saf_delay)
        tot0, _ = mandate_shares(P, years)
        tot = np.where(years - saf_delay < years[0], tot0[0], tot)
        r_p = saf_r if saf_r_ptl is None else saf_r_ptl
        Fbasis = fuel.sum(0) if saf_fixed_fuel is None else saf_fixed_fuel
        s_ptl = r_p * syn * Fbasis
        s_bio = saf_r * (tot - syn) * Fbasis
    else:
        s_ptl = np.zeros(T)
        s_bio = P["saf.base_share_2024"] * fuel.sum(0)

    return Result(years, fuel, zep_eq, gross, atm, phi, pi, s_bio, s_ptl, kappa,
                  P.e_wtw, P["saf.theta_bio"], P["saf.theta_ptl"], paths)


def run_scenario(P: Params, base: Base, key: str, **kw) -> Result:
    return run(P, base, SCENARIOS[key].levers, **kw)


# ------------------------------------------------------------------ attribution
def shapley(P: Params, base: Base, levers: tuple = LEVERS, metric: str = "2050") -> pd.DataFrame:
    """Shapley attribution of the reduction from S0 to the full lever set, by lever and band.

    metric: '2050' (emissions in 2050) or 'cum' (cumulative 2025-2050). Values in Mt CO2e,
    positive = reduction. Contributions sum exactly to the total reduction in each band."""
    n = len(levers)
    cache: dict[frozenset, np.ndarray] = {}

    def value(S: frozenset) -> np.ndarray:
        if S not in cache:
            eb = run(P, base, S).emissions_by_band
            cache[S] = eb[:, -1] if metric == "2050" else eb[:, 1:].sum(axis=1)
        return cache[S]

    out = {}
    for i, lev in enumerate(levers):
        others = [l for l in levers if l != lev]
        phi = np.zeros(4)
        for k in range(n):
            for comb in itertools.combinations(others, k):
                S = frozenset(comb)
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                phi += w * (value(S) - value(S | {lev}))
        out[lev] = phi / 1e6
    df = pd.DataFrame(out, index=P["bands.short"])
    return df
