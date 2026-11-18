## Apollo 8-inspired hardware budget

The capture simulator models the Apollo 8 command and service module (CSM) and its service propulsion system (SPS).
SI conversion uses 0.45359237 kg/lb and 4.4482216152605 N/lbf.

| Quantity | Adopted value / derivation |
|---|---|
| Parking-orbit stack mass | 283,213 lb = 128,463 kg approximately |
| CSM wet mass | (12,392 + 51,258) lb = 28,871 kg approximately |
| Allocated departure propellant | (283,213 - 122,380) lb = 72,953 kg approximately |
| Discarded inert/residual assembly | 122,380 lb minus CSM wet mass |
| SPS loaded propellant | (15,730.9 + 25,105.1) lb = 18,523 kg approximately |
| CSM inert mass | CSM wet mass minus SPS loaded propellant |
| J-2 model | 200,000 lbf, 418 s specific impulse, fixed thrust |
| SPS model | 20,396 lbf, 313 s specific impulse, fixed thrust |

The departure allocation uses the planned mass loss from parking orbit to injection. The remaining mass outside the CSM is treated as discardable hardware and residual propellant. The model uses a fixed nominal J-2 rating.

Sources (PDF copies at the repository root: `apollo8_source.pdf` and `apollo8_report.pdf`):

- [Apollo 8 press kit](https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A08_PressKit.pdf), printed pp. 33--34, 40--41: CM/SM masses, stack masses, engine ratings.
- [Apollo 8 mission report](https://ntrs.nasa.gov/citations/19700033031), pp. 6-81, 6-99: measured SPS performance and propellant loading.
- [NASA Saturn V engine systems](https://ntrs.nasa.gov/api/citations/20240002646/downloads/05-LV_Loads_Sizing.pdf): nominal J-2 rating and specific impulse.


Capture flights start with 50–100% of the listed SPS propellant load, all modeled as usable.
