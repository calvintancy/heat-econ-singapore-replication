# Earlier heat and electricity-demand project

Located and consulted on 10 November 2026:

- Source: `/Users/calvintan/.openclaw/workspace/projects/ema_empirical_note_aea_2026-03-08`
- Pipeline: `scripts/build_ema_empirical_note.py`, `scripts/build_ema_empirical_note_legacy.py`
- Latest robustness code: `scripts/build_aea_robustness_suite.py`
- [July robustness report](/Users/calvintan/.openclaw/workspace/projects/ema_empirical_note_aea_2026-03-08/outputs/annex/aea_robustness_suite_20260705/aea_robustness_suite_report.md)
- Data-construction audit: `DATA_COMPILATION_AUDIT.md` and `PIPELINE_FILE_MAP.md`
- Public snapshot: `/Users/calvintan/.openclaw/workspace/projects/ema_empirical_note_public`
- Weather inputs: `/Users/calvintan/.openclaw/workspace/data/ema_master`

The preferred model regresses actual EMA system demand in MW on 1°C temperature bins (28–29°C omitted), rainfall, wind speed, relative humidity and PM2.5. Fixed effects are half-hour-of-day, day-of-week, year-month, public holiday and school holiday. Standard errors cluster by calendar date. The canonical complete sample contains 144,901 half-hours, from 1 January 2017 at 08:00 to 31 December 2025 at 23:30.

The July annex reports a reproduced +1°C bin-shift demand translation of 95.041471 MW. This is a reference finding, not a result of the new VA analysis. An older Downloads HTML report uses an earlier hourly/NEM setup, so it is not the preferred reference.

The canonical weather construction averages raw readings within station × half-hour and then equally across active stations. Use `*_halfhour_station_avg_2016_2025.csv` for any later short-sample comparison, alongside station-count and quality files. The older hourly weather master interpolates observations and should not be substituted without documenting that transformation.

Wind data are in knots; older prose contains an m/s label error. Convert by 0.514444 to m/s if needed. The half-hour rainfall variable is an average reported gauge reading, which should not be treated as a validated quarterly rainfall total. The new project instead uses official monthly Changi rainfall totals.

No earlier VA/GDP growth regression was found. The earlier demand study is a national time series, rather than a sector panel. The new project preserves its weather-control/calendar-FE approach, with deliberate changes:

| Earlier demand study | First VA specification |
| --- | --- |
| National half-hourly demand level | Sectoral quarterly real VA log growth |
| Temperature bins | Linear quarterly mean air temperature |
| Active-station national averages | Changi monthly climate series, aggregated to quarters |
| Rainfall, RH, wind, PM2.5 | Quarterly rainfall and RH, available over the long sample |
| Fine calendar and year-month FE | Sector × quarter-of-year and year FE |
| Date-clustered SE | Quarterly Driscoll–Kraay SE; quarter-cluster comparator |

Wind and PM2.5 can be considered in a separate 2017–2025 comparison, but that would leave just 36 national quarters and change the exposure measurement. Full year-quarter FE absorb common national temperature. A model with those FE would require sector-specific heat exposures or identify temperature interactions relative to a reference sector.
