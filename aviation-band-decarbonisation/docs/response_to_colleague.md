# How the revised model answers the colleague's review

Section and equation numbers refer to the revised `paper/main.pdf`; the headings use the old
equation numbers the colleague referred to.
Numbers quoted here are from the default run; the authoritative values are in
`paper/generated/numbers.tex` and change if `config/parameters.yaml` changes.

## 1. ReFuelEU share in old Eq. (5): "≤" or "≥"? (now Eq. 4)

The previous "σ ≤ 0.70" was wrong. ReFuelEU sets **minimum** shares, applied fleet-wide to fuel
suppliers (volume-weighted across Union airports until 2034), and they are stepwise by period.
The constraint is now (Eq. 4)

    σ_min,t · ΣF ≤ S_bio + S_syn ≤ S̄_t,   S_syn ≥ σ_syn,t · ΣF

so the mandate is a lower bound and the binding upper bound is physical supply S̄_t.
Two consequences are in the paper: SAF allocation across bands is CO₂-neutral for a given quantity
(Eq. 5), and under binding supply every tonne of fuel saved displaces fossil kerosene.

Code: `system.mandate_shares`, `system.run`; tests `test_mandate_is_minimum_and_stepwise`,
`test_saf_allocation_neutral`.

## 2. Combustion vs life-cycle factors in old Eq. (1); ASK vs RPK in old Eq. (3) (now Eqs. 1–3)

Emissions are now **well-to-wake throughout**: the fossil factor is e = 89 g CO₂e/MJ × 43.15 MJ/kg
≈ 3.84 t CO₂e/t, and θ multiplies e, never the combustion factor 3.16. ZEP gets the same boundary
through an upstream factor κ_t; with κ = 0, ZEP would get a full life-cycle credit that SAF does not.
The index is invariant to the fossil factor (test `test_index_invariant_to_fossil_factor`), so the
earlier indexed trajectories were not affected; absolute tonnes were.

Eq. (3) is split into a per-ASK and a per-RPK indicator; load factor enters only the latter.
At flight level load factor raises fuel per flight (payload mass) and lowers fuel per passenger;
the regression reproduces both signs (η = +0.15 per flight, −1.10 per RPK; test `test_load_factor_signs`).

## 3. "Long-haul renewal maximises abatement per unit of capital"

The claim is now computed, not asserted. Eq. (12) gives the abatement cost of advancing one
retirement by band (capital recovery of new minus residual value, net of fuel savings, per tonne
CO₂e), following Schäfer et al. (2016) and Salgas et al. (2025). With list prices × market ratio,
utilisation, generation step and 2024 ReFuelEU reference fuel prices (all in Appendix A):

* long-haul ≈ EUR 10/t, other bands ≈ EUR 260–270/t; abatement ≈ 55 kt/yr per long-haul aircraft vs 5–10 kt
* long-haul remains the cheapest renewal option in every sensitivity variation (aircraft price ±20 %,
  fuel price 550–950 EUR/t, …)

So the original claim holds under the stated inputs, but it now rests on a table (Table 3) and a
sensitivity analysis (Fig. 6) rather than on intuition.

## 4. Common age coefficient in old Eq. (8) vs "steeper long-haul gradients" (now Eq. 7)

In a log model a common γ means the same **percentage** effect per year in every band; the
**absolute** effect is γ·E and is therefore far larger at long range. The regression now has
band-specific slopes γ_d and a Wald test of γ_4 = γ_1. On the generated data γ_4 > γ_1, but only
because the assumed generation step is larger for wide-bodies (22 % vs 15 %), so the paper reports
this as reproducing an input. The robust statement is the absolute one: one year of age adds about
90× more CO₂ to a long-haul flight than to a sub-500 km flight. The earlier text attributed the
gradient to an age term in Eq. (2) that did not exist; that is removed. Aircraft age now enters the
system model through an explicit fleet vintage model (Eq. 6, `fleet.py`).

## The proposed extension: SAF supply delays and minimum corrective action

Implemented as Sections 2.7–2.8 and 4.3–4.4:

* **Disruption** (Eq. 9) S_t(δ, r) = r · σ_min,t−δ · F_nom,t, with r ∈ [0.57, 0.95] from Martulli et al. (2025).
* **Tolerance**: cumulative and 2050 excess over (δ, r) (Fig. 4) and a closed-form critical
  realisation rate r* (Eq. 10), verified against a numerical root (`test_rstar_closed_form_matches_root`).
  Against a pathway's own target r* depends only on θσ₂₀₅₀; efficiency adds robustness only against an
  external target (Costa-Alves et al. 2026 framing).
* **Minimum corrective action**: LP (Eq. 11) with three levers by band and year: ATM measures,
  early retirement (delivery-slot limited) and alternative SAF; residuals flag the uncompensable part.
* **Headline results** (default parameters): a one-year delay raises 2050 emissions by ~52 % but
  cumulative emissions by ~3 % (mandate steps 42 → 70 % in 2050). After an 18 % shortfall, restoring
  the cumulative budget is a net saving of ~EUR 7 bn (ATM measures + early retirement of ~300 long-haul
  aircraft); also restoring the 2050 level costs ~EUR 17 bn and still leaves 3 % of the 2050 gap. With a
  three-year delay most of the 2050 gap is uncompensable. S3, which pre-commits delivery slots to its own
  accelerated renewal, leaves 50 % of its 2050 gap open even without delay.

## Other corrections found during the revision

* Band sizes in the old Table 2 did not match the old F/Tukey statistics (they matched CO₂ shares).
  The ANOVA on synthetic data is dropped; the new generator imposes flight shares and fits CO₂ shares.
* Fig. 4 of the old version could not be produced by its Table 3. All numbers now come from code.
* S3's lead over S2 was attributed to ZEP; Shapley attribution shows ZEP contributes ~3 % of the 2050
  reduction.
* ATM is now an explicit lever (taxi, ASMA, en-route within European airspace), calibrated to the
  9 % benefit pool of PRR 2024, as required by the special issue.
