"""Band-level fleet vintage model.

Each band holds a fleet described by annual cohorts (age 0 .. L-1). Cohorts age,
reach retirement at age L, and are replaced by deliveries whose fuel intensity
improves by a fixed annual rate. Accelerated renewal retires the oldest cohorts
early, limited by the delivery slots available each year. The base-year fleet
is the one the flight generator samples (uniform ages, class-specific
generation step), so the two models are consistent by construction.

Intensities are expressed as multipliers on the fuel burn of a new-generation
aircraft of the base year; the fleet index Phi_{d,t} is the km-weighted mean
multiplier divided by its base-year value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import Params


@dataclass
class BandFleet:
    km_weights: np.ndarray       # km share of each class in the band (RG, NB, WB)
    util_km: float               # band-average utilisation (km per aircraft-year)
    m_base: np.ndarray           # multiplier by base-year age 0..L-1
    dphi: float                  # km-weighted new-generation fuel reduction


def band_fleets(P: Params, km_by_band_class: np.ndarray) -> list[BandFleet]:
    """km_by_band_class: 4 x 3 array of base-year km by band and class (from the flight sample)."""
    L = int(P["fleet.age_max_years"])
    ages = np.arange(L) + 0.5
    deter = P["fleet.deterioration_per_year"]
    util = P.arr("aircraft_classes.utilisation_km_per_year")
    dphi = P.arr("aircraft_classes.newgen_fuel_reduction")
    ng = P.arr("aircraft_classes.newgen_max_age_years")
    out = []
    for d in range(4):
        w = km_by_band_class[d] / km_by_band_class[d].sum()
        m_class = np.array([np.where(ages <= ng[c], 1.0, 1.0 / (1 - dphi[c])) * (1 + deter * ages)
                            for c in range(3)])
        m = w @ m_class
        u = 1.0 / np.sum(w / util)          # aircraft count = sum_c km_c / U_c
        out.append(BandFleet(w, float(u), m, float(w @ dphi)))
    return out


@dataclass
class FleetPath:
    phi_index: np.ndarray        # Phi_{d,t} / Phi_{d,base}
    m_mean: np.ndarray           # km-weighted mean multiplier
    m_oldest: np.ndarray         # multiplier of the oldest cohort still flying
    m_new: np.ndarray            # multiplier of that year's deliveries
    aircraft: np.ndarray         # fleet size
    early_retired: np.ndarray    # early retirements used by the scenario itself
    slots: np.ndarray            # slot cap (aircraft)


def simulate(P: Params, bf: BandFleet, km_path: np.ndarray, retire_age: int | None = None,
             accelerate: bool = False) -> FleetPath:
    """Vintage simulation for one band.

    km_path   : annual km flown by the band (base year first)
    retire_age: target retirement age under accelerated renewal (default: natural age)
    accelerate: if True, cohorts older than retire_age are retired early, up to the slot cap
    """
    L = int(P["fleet.age_max_years"])
    target = L if retire_age is None else int(retire_age)
    tau = P["fleet.new_delivery_improvement_per_year"]
    deter = P["fleet.deterioration_per_year"]
    slots_share = P["corrective.slots_share_of_fleet"]
    T = len(km_path)

    # cohorts tracked as km capacity shares; index = age
    cap = np.full(L, km_path[0] / L)
    mult = bf.m_base.copy()                       # multiplier of each cohort (age-dependent part included)
    phi = np.zeros(T); m_mean = np.zeros(T); m_old = np.zeros(T); m_new = np.zeros(T)
    ac = np.zeros(T); early = np.zeros(T); slots = np.zeros(T)
    m_mean[0] = (cap * mult).sum() / cap.sum()
    m_old[0] = mult[-1]
    m_new[0] = 1.0
    ac[0] = km_path[0] / bf.util_km
    slots[0] = slots_share * ac[0]
    for t in range(1, T):
        # age by one year: deterioration on surviving cohorts, oldest cohort retires naturally
        retiring = cap[-1]
        cap = np.concatenate([[0.0], cap[:-1]])
        mult = np.concatenate([[1.0], mult[:-1] * (1 + deter)])
        # accelerated renewal: retire cohorts aged >= target, oldest first, within slot cap
        slot_km = slots_share * (cap.sum() / bf.util_km) * bf.util_km
        used = 0.0
        if accelerate and target < L:
            for a in range(L - 1, target - 1, -1):
                take = min(cap[a], slot_km - used)
                if take <= 0:
                    break
                cap[a] -= take
                used += take
        # deliveries fill natural + early retirements + growth
        new_mult = (1 - tau) ** t
        cap[0] = max(km_path[t] - cap.sum(), 0.0)
        mult[0] = new_mult
        m_mean[t] = (cap * mult).sum() / cap.sum()
        alive = np.nonzero(cap > 1e-9)[0]
        m_old[t] = mult[alive[-1]]
        m_new[t] = new_mult
        ac[t] = km_path[t] / bf.util_km
        early[t] = used / bf.util_km
        slots[t] = slots_share * ac[t]
    phi = m_mean / m_mean[0]
    return FleetPath(phi, m_mean, m_old, m_new, ac, early, slots)
