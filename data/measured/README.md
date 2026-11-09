# Observed flight data (optional)

Place a file `flights.csv` here and run `python scripts/run_all.py --measured`.
The file is git-ignored by default; commit it only if its licence allows redistribution.

| column | unit | meaning |
|---|---|---|
| `band` | 0–3 | distance band: 0 = 0–500 km, 1 = 500–1500, 2 = 1500–4000, 3 = >4000 |
| `aclass` | 0–2 | aircraft class: 0 regional, 1 narrow-body, 2 wide-body |
| `distance_km` | km | great-circle stage length |
| `seats` | – | installed seats |
| `load_factor` | 0–1 | passengers / seats |
| `age` | years | aircraft age at the flight date |
| `fuel_t` | t | gate-to-gate fuel burned |
| `fuel_atm_excess_t` | t | ATM-related excess fuel (set to 0 if unknown; the ATM analysis then uses the model estimate) |

Minimum for the band statistics: `band`, `distance_km`, `seats`, `load_factor`, `fuel_t`.
Age and class are needed for the regression of Eq. (8).
