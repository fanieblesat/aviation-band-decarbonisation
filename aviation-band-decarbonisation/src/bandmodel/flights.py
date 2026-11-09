"""Flight-level emission model: physically based generator, calibration and band statistics.

Emissions of a flight are computed from its aircraft class, stage length, age
(generation and deterioration), load factor and ATM inefficiency. Band
differences therefore follow from physics and fleet composition rather than
from assumed band means. Three class-specific cruise fuel scalings are fitted so
that the band CO2 shares reproduce EUROCONTROL's 2019 shares; flight shares are
imposed by sampling. Seat-km and per-passenger-km patterns are then checked
against independent sources that were not used in the fit.

If measured flight data is available (data/measured/flights.csv with the columns
of `FLIGHT_COLUMNS`), `load_measured` replaces the generator and every downstream
statistic is computed on the observations instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .params import ROOT, Params

FLIGHT_COLUMNS = ["band", "aclass", "distance_km", "seats", "load_factor", "age",
                  "fuel_t", "fuel_atm_excess_t"]


@dataclass
class Calibration:
    cruise_scale: np.ndarray        # multiplicative factor on cruise fuel per class
    other_enroute_share: float      # vertical/other inefficiency calibrated to the PRR benefit pool
    co2_share_fit: np.ndarray
    residual: float


# --------------------------------------------------------------------------- physics
def flight_fuel(P: Params, aclass: np.ndarray, d: np.ndarray, age: np.ndarray,
                lf: np.ndarray, scale: np.ndarray, other_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Gate-to-gate fuel (t) and its ATM-related excess (t) for each flight.

    efficient fuel = LTO + cruise(d) * generation * deterioration * payload
    excess         = taxi + ASMA + (KEA + other) * cruise fuel within European airspace
    """
    lto = P.arr("aircraft_classes.lto_fuel_t")[aclass]
    c = P.arr("aircraft_classes.cruise_fuel_kg_per_km")[aclass] * scale[aclass] / 1000.0
    curv = P.arr("aircraft_classes.weight_curvature_per_km")[aclass]
    dphi = P.arr("aircraft_classes.newgen_fuel_reduction")[aclass]
    ng_age = P.arr("aircraft_classes.newgen_max_age_years")[aclass]
    deter = P["fleet.deterioration_per_year"]
    lf0 = P["fleet.load_factor_mean"]
    psi = P["fleet.payload_fuel_sensitivity"]

    # cruise fuel of a new-generation aircraft is the reference; older generation burns more
    gen = np.where(age <= ng_age, 1.0, 1.0 / (1.0 - dphi))
    mult = gen * (1.0 + deter * age) * (1.0 + psi * (lf - lf0))
    cruise = c * d * (1.0 + curv * d) * mult
    efficient = lto * mult + cruise

    taxi = P["atm.additional_taxi_out_min"] * P.arr("aircraft_classes.taxi_fuel_kg_per_min")[aclass] / 1000.0
    asma = P["atm.additional_asma_min"] * P.arr("aircraft_classes.hold_fuel_kg_per_min")[aclass] / 1000.0
    d_eu = np.minimum(d, P["atm.european_airspace_km"])
    cruise_eu = cruise * d_eu / d
    excess = taxi + asma + (P["atm.kea_excess_distance"] + other_share) * cruise_eu
    return efficient + excess, excess


def sample_flights(P: Params, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample flights with band shares, stage lengths, classes, ages and load factors (no fuel yet)."""
    shares = P.arr("bands.flight_share_2019")
    band = rng.choice(4, size=n, p=shares / shares.sum())
    tri = np.asarray(P["bands.stage_triangular_km"], dtype=float)
    d = rng.triangular(tri[band, 0], tri[band, 1], tri[band, 2])
    mix = np.asarray(P["band_class_mix"], dtype=float)
    u = rng.random(n)
    cum = np.cumsum(mix, axis=1)[band]
    aclass = (u[:, None] > cum).sum(axis=1)
    age = rng.uniform(0, P["fleet.age_max_years"], size=n)
    m = P["fleet.load_factor_mean"]
    k = 40.0  # beta concentration: sd about 0.06
    lf = rng.beta(m * k, (1 - m) * k, size=n)
    seats = P.arr("aircraft_classes.seats")[aclass]
    return pd.DataFrame({"band": band, "aclass": aclass, "distance_km": d, "seats": seats,
                         "load_factor": lf, "age": age})


def calibrate(P: Params, df: pd.DataFrame) -> Calibration:
    """Fit class cruise scalings to band CO2 shares, and the 'other' en-route
    inefficiency to the PRR benefit pool of intra-European flights (< 4000 km)."""
    target = P.arr("bands.co2_share_2019")
    a = df["aclass"].to_numpy()
    d = df["distance_km"].to_numpy()
    age = df["age"].to_numpy()
    lf = df["load_factor"].to_numpy()
    band = df["band"].to_numpy()
    pool_target = P["atm.benefit_pool_intra_europe"]

    def shares(scale: np.ndarray, other: float) -> tuple[np.ndarray, float]:
        fuel, exc = flight_fuel(P, a, d, age, lf, scale, other)
        by_band = np.bincount(band, weights=fuel, minlength=4)
        intra = band < 3
        pool = exc[intra].sum() / fuel[intra].sum()
        return by_band / by_band.sum(), pool

    # Shares are ratios, so a common factor on all classes is not identified: the
    # narrow-body scaling is pinned at 1 (engineering value) and anchors the level.
    def to_scale(x: np.ndarray) -> np.ndarray:
        return np.array([np.exp(x[0]), 1.0, np.exp(x[1])])

    def resid(x: np.ndarray) -> np.ndarray:
        s, pool = shares(to_scale(x), x[2])
        return np.concatenate([(s - target) / target, [(pool - pool_target) / pool_target]])

    sol = least_squares(resid, x0=np.array([0.0, 0.0, 0.02]),
                        bounds=([-1.5, -1.5, 0.0], [1.5, 1.5, 0.2]))
    scale = to_scale(sol.x)
    s, _ = shares(scale, sol.x[2])
    return Calibration(scale, float(sol.x[2]), s, float(np.abs(sol.fun).max()))


def generate(P: Params, n: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, Calibration]:
    rng = np.random.default_rng(P["meta.seed"] if seed is None else seed)
    df = sample_flights(P, n or int(P["meta.n_flights_sample"]), rng)
    cal = calibrate(P, df)
    fuel, exc = flight_fuel(P, df["aclass"].to_numpy(), df["distance_km"].to_numpy(),
                            df["age"].to_numpy(), df["load_factor"].to_numpy(),
                            cal.cruise_scale, cal.other_enroute_share)
    df["fuel_t"] = fuel
    df["fuel_atm_excess_t"] = exc
    return enrich(P, df), cal


def load_measured(P: Params, path: str | Path | None = None) -> pd.DataFrame:
    """Load observed flights (same schema) – replaces the generator when available."""
    path = Path(path) if path else ROOT / "data" / "measured" / "flights.csv"
    df = pd.read_csv(path)
    missing = set(FLIGHT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"measured data lacks columns: {sorted(missing)}")
    return enrich(P, df)


def enrich(P: Params, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["co2_ttw_t"] = df["fuel_t"] * P.e_ttw
    df["ask"] = df["seats"] * df["distance_km"]
    df["rpk"] = df["ask"] * df["load_factor"]
    df["g_per_ask"] = df["co2_ttw_t"] * 1e6 / df["ask"]
    df["g_per_rpk"] = df["co2_ttw_t"] * 1e6 / df["rpk"]
    df["ln_co2"] = np.log(df["co2_ttw_t"])
    df["atm_share"] = df["fuel_atm_excess_t"] / df["fuel_t"]
    return df


# --------------------------------------------------------------------------- statistics
def band_table(P: Params, df: pd.DataFrame) -> pd.DataFrame:
    """Per-band structure: shares, per-flight, per-ASK and per-RPK intensity, ATM excess."""
    g = df.groupby("band")
    tot_f = len(df)
    tot_c = df["co2_ttw_t"].sum()
    out = pd.DataFrame({
        "flight_share": g.size() / tot_f,
        "co2_share": g["co2_ttw_t"].sum() / tot_c,
        "seat_share": g["seats"].sum() / df["seats"].sum(),
        "seatkm_share": g["ask"].sum() / df["ask"].sum(),
        "mean_distance_km": g["distance_km"].mean(),
        "co2_per_flight_t": g["co2_ttw_t"].mean(),
        "mean_ln_co2": g["ln_co2"].mean(),
        "g_per_ask": g["co2_ttw_t"].sum() * 1e6 / g["ask"].sum(),
        "g_per_rpk": g["co2_ttw_t"].sum() * 1e6 / g["rpk"].sum(),
        "atm_excess_share": g["fuel_atm_excess_t"].sum() / g["fuel_t"].sum(),
    })
    out.index = P["bands.short"]
    return out


def age_regression(df: pd.DataFrame) -> dict:
    """Eq. (7): ln(CO2 per flight) on band intercepts and band-specific age slopes, plus
    ln(CO2 per RPK) on load factor. Returns coefficients, SEs and the Wald test gamma_4 = gamma_1."""
    import statsmodels.formula.api as smf

    d = df.copy()
    d["band"] = d["band"].astype("category")
    d["ln_rpk"] = np.log(d["g_per_rpk"])
    m1 = smf.ols("ln_co2 ~ C(band) + C(band):age + np.log(distance_km) + C(aclass)", data=d).fit(cov_type="HC1")
    names = [n for n in m1.params.index if ":age" in n]
    gam = m1.params[names].to_numpy()
    se = m1.bse[names].to_numpy()
    wald = m1.wald_test(f"{names[3]} - {names[0]} = 0", scalar=True)
    m2 = smf.ols("ln_rpk ~ load_factor + C(band) + age + C(aclass)", data=d).fit(cov_type="HC1")
    m3 = smf.ols("ln_co2 ~ load_factor + C(band) + age + C(aclass) + np.log(distance_km)", data=d).fit(cov_type="HC1")
    return {"gamma": gam, "gamma_se": se, "wald_F": float(wald.statistic), "wald_p": float(wald.pvalue),
            "eta_rpk": float(m2.params["load_factor"]), "eta_rpk_se": float(m2.bse["load_factor"]),
            "eta_flight": float(m3.params["load_factor"]), "eta_flight_se": float(m3.bse["load_factor"]),
            "r2": float(m1.rsquared), "n": int(m1.nobs)}
