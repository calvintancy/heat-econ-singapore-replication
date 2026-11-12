# Current native-Wald presentation coverage and historical boundaries

This is a source/display map for the new presentation layer, not a change to any regression, covariance, sample, coefficient t-test, old joint CSV or release. The current master contains 21 embedded report chapters plus Overview and Methods. The new staged master adds a native-Wald tab and uses separate qualified copies of those 21 chapters.

## Numerical routes in the preserved generation

| Producer / function | Consumers and saved results | Existing reference |
|---|---|---|
| `heterogeneity.py::wald` (154) | Broad H1/H2/H3/R1/H3X; `granular_interactions.py`; `covid_sector.py`; six `outputs/heterogeneity/*/{weather,full}/joint_tests.csv`, nine detailed-interaction files, paired sector joint CSV | W/q with F(q,T−1), rank guard |
| `annual_analysis.py::_wald` (374) | `outputs/annual/joint_tests.csv` | Equal-sector slopes, one-temperature rank cap, F(q,T−1) |
| `clustered_sector_sensitivity.py::joint_test` (104) | Raw 90-fit /270-row sensitivity; qualified viewer retains calendar comparisons and suppresses unsupported sector-CR1 significance | F(q,G−1) or F(q,T−1), HC0/rank guard |
| `visitor_analysis.py::Analysis.profile` (342) | `outputs/visitor_arrivals/joint_tests.json` (16 rows) and `designs/profiles__{group}__{band}.npz` | JSON `statistic` is F, not W; profile covariance already G×G |
| `crisis_sensitivity.py::profile_tests` (283) | Subannual crisis, complete-GVA five-control extension via `complete_va_controls.py` | Equal slopes, custom CR1/DK covariance, F |
| `annual_crisis_sensitivity.py::joint` (95) | Annual crisis joint CSV | Equal slopes, F(q,T−1) |
| `literature_sensitivity.py::test_record` (211) | Subannual lag/shape/equality/all-zero tests; seasonal caller; long two-channel-span supplement | W/q, custom covariance and F; original seasonal/subannual ranks subsequently reassessed |
| `literature_rank_correction.py::assess` (28) | Final subannual and annual-seasonal `tests.csv`; native-stage archives remain historical | Same W, actual rank and universal T−1/G−1 score cap, F |
| `annual_literature_sensitivity.py::wald` (98) | Annual DJO/BHM explicit restrictions | General Rβ=0, F with recorded covariance-specific df |
| `literature_full_controls.py::wald` (135) | Full wind/PM tests and `literature_full_controls_restrictions.py` span supplement; new-arrivals monthly wrapper | General Rβ=0, rank/HC0 guard, F |
| `visitor_rigour_monthly.py::wald` (279), `visitor_origin_year.py` aliases | Monthly and origin-year `tests.csv` | Calls full-control restriction kernel; shared native coefficient basis / preserved covariance |
| `visitor_rigour_companions.py::joint` (142) | Annual `tests.csv` plus alias `joint_tests.csv`; PPML `tests.csv` | Preserved covariance and F, including explicit unavailable rows |
| `verify_stata_key_results.py` joint block (around 200), `verify_stata_visitor_rigour.py` (around 210) | Stata heat and arrival joint CSVs | Python restriction postprocessing on native Stata/Mata arrays; not a newly executed Stata `test` command |
| `outcome_analysis.py::run_context`/YoY block (around 178), `aggregate_va.py` YoY block (around 188) | Eight temperature-sum JSONs; six contain a nested sector-cluster result, yielding 14 covariance rows | One-constraint two-sided t; equivalent historical F(1,df), not all-weather joint restrictions |
| `verify_pyfixest_migration.py` one-constraint total-effect audit | `sector_total_effect_null_checks.csv` / original coefficient-source locator | βT+δᵢ=0; mathematical t²/F1 audit. Keep coefficient t rows unchanged and show native χ² audit separately |

Native PyFixest 0.70.0 `Feols.wald_test` delegates to `pyfixest.estimation.post_estimation.wald.wald_test`. The pure package kernel accepts explicit β,V,R,r and `distribution="chi2"`, returning W, W/q and χ² p. The model method only retains F for its default full identity/all-zero restriction; general explicit R falls back to χ². Do not call the model default without R to stand in for equal sector slopes: that would test a different null, potentially including nuisance coefficients.

“Native Wald” must mean the package test kernel, not a claim that the saved custom CR1/HAC covariance became PyFixest's default VCE. A one-constraint χ² p differs from the retained finite-df t p even when the null and W=t² agree. Both must be labelled.

## Current chapter source map

`native_wald_report.py::CHAPTER_SOURCE_RULES` is the executable map. All rows also remain browsable in the global native-Wald tab and audit chapter.

| Chapter key | Current source routes |
|---|---|
| `complete-crisis` | `crisis_sensitivity/joint_tests.csv`, `crisis_sensitivity/annual/joint_tests.csv`, `complete_va_controls/joint_tests.csv` |
| `literature-models` | Root/annual/annual_seasonal/full_controls final `tests.csv`; root and full-controls `two_channel_span_tests.csv` |
| `referee-audit` | All current overlay rows; original reviewer quotes and calibration evidence remain historical |
| `stata-check` | `stata_key_results/joint_tests.csv` |
| `visitor-rigour` | Four visitor-rigour branches' canonical `tests.csv`; `stata_key_results/arrivals/joint_tests.csv` |
| `covid` | `covid_comparison/sector/joint_tests.csv` |
| `annual` | `annual/joint_tests.csv` |
| `aggregate` | Two aggregate-VA temperature-difference-restriction JSONs |
| `growth-levels` | Six VA/monthly-IIP/quarterly-IIP temperature-difference-restriction JSONs |
| `sector-va`, `iip` | Their original total-effect coefficient-audit source locators, if supplied; no new joint null is invented for ordinary coefficient tables |
| `granular-va` | `granular_interactions/va-detailed/*/joint_tests.csv` and corresponding total-effect audit locators |
| `manufacturing` | Monthly/quarterly detailed-IIP interaction joint files and corresponding audit locators |
| `visitors` | Original 16-row `visitor_arrivals/joint_tests.json`; crisis JSON is empty and inventoried |
| `heterogeneity` | Six broad heterogeneity joint CSVs and matching original coefficient-audit locators |
| `cluster-sensitivity` | Raw `clustered_sector_sensitivity/joint_tests.csv`; unsupported guards remain unavailable |
| `new-stationarity`, `stationarity`, `wind-pm`, `tables`, `reconciliation` | Explicit no-joint message unless a collector supplies a declared chapter alias. ADF/KPSS tests and ordinary coefficient t-tests are not converted to Wald tests |

Each current widget has separate family/context/outcome/equation/sample/covariance/unit/hypothesis selectors. It explicitly does not synchronize with the preserved coefficient selectors. Every row repeats the active model/sample/covariance. Selected CSV export carries source identity. Legacy joint containers are collapsed where structurally identifiable; old table-stat rows receive historical-F labels; the whole preserved chapter is visibly qualified. Old standalone files are untouched.

## Builders and displays that must read current overlay fields

- `heterogeneity_report.py::sector_table/result_panel`: custom native-table statistics, prose lead, equality table and excluded-sample readout.
- `granular_report.py::interaction_native/interaction_results`: custom joint p row, F/q/df box and badges.
- `annual_report.py::joint_readout/build_views`: F label and generated HTML held in selector payload.
- `covid_report.py::build_covid_report`: `joint_tests` payload and inline `joint-readout` F/p table.
- `visitor_report.py::build_report`/inline `drawProfile`: original profile equality p and source JSON.
- `crisis_report.py::build` plus `templates/crisis_sensitivity.html`: joint payload and F table.
- `literature_report.py::build/normalize` plus `templates/literature_sensitivity.html`: four branch test payloads, span supplement, F column, alias and conditional-shape wording.
- `visitor_rigour_report.py::load_branch/build_visitor_rigour_report` plus `templates/visitor_rigour.html`: four family test payloads, Stata comparison/source downloads and nominal p wording.
- `qualified_clustered_sector_sensitivity.py`: calendar equality readout; raw sensitivity source's historical F table stays archival.
- `stata_key_report.py`: joint CSV is principally a download rather than an on-page joint table. Its new widget supplies the current restriction readout without rewriting native esttab output. Arrival Stata joint CSV similarly belongs in the visitor-rigour widget.
- `outcome_report.py::yoy_weather` and `aggregate_va_report.py::build_aggregate_va_report`: old temperature-only t p paragraphs must be explicitly historical comparisons to the current q1 χ² audit, without changing coefficient t-tests.
- `aea_audit_report.py`: quoted old F/null-simulation counts and review reports must stay tied to their original reference. A new χ² label alone cannot convert an old F simulation into a newly executed calibration study.
- `combined_report.py::REPORTS/embedded_report/build_combined_report`: old source stays immutable. New staging replaces its embedded chapter payloads with qualified copies and adds a native-Wald tab. Overview coefficient t p-values remain unchanged.

The actual implementation writes new files rather than modifying any builder above. The archived builder map identifies every old readout that the new copy qualifies.

## Immutable evidence and new proof

Do not rewrite `data/reference_before_pyfixest`, `data/reference_before_nsa_inputs`, the original migration/input-reference JSONs, paired-COVID original-input checkpoint, old source seals/releases, original independent-verifier expectations or the arrivals review baseline. Many verifiers explicitly assert the old F p-values. They remain valid historical acceptance checks, not tests of the new primary display.

The immutable arrivals baseline checks 534 semantic files/8,338 arrays and original source AST bindings, including old test CSV p-values. Keeping the new native-Wald layer separate lets that exact equivalence proof remain meaningful. Likewise preserve old calibration draws and their F/t reference: their evidence is not a current χ² calibration certificate. New current numerical, display, browser and source-seal receipts must be separate and clearly bind the new overlay.

`outputs/native_wald/predecessor_manifest.json` binds the byte-exact 21 standalone reports and predecessor master. The new builder writes `outputs/native_wald/reports/*`, `native_wald.html`, `current_master.html` and `report_manifest.json`; the new presentation verifier writes `report_verification.json`. No integration overwrites the master before the root integration step for the verified staged artifact. All coefficient/V/sample arrays remain unchanged, and χ² is not represented as a calibration cure.

The optional UI trial uses `uv run --locked python scripts/verify_native_wald_report.py --prototype`, writing `report_trial_verification.json`; it cannot stand in for the normal current-source numerical/presentation gate. `uv run --locked --with playwright==1.65.0 python scripts/verify_native_wald_browser.py --staged` opens an isolated headless Google Chrome profile and writes `browser/staged_verification.json`. After root integrates the verified staged master, the same browser command without `--staged` requires the integrated and staged master hashes to match and writes the distinct current gate `browser/verification.json`. Invoke the normal source-backed gate without `--prototype`. Both gates invalidate previous current proof before reading inputs, archive prior receipts, pin tested source/artifact bytes at entry, and require identical bytes after the runtime checks. Node VM control testing is identified separately from actual Chrome.

Dynamic/AR level-channel, equal-growth and endpoint-plus-sum span hypotheses are labelled as conditional direct-weather coefficient shapes with growth lags held fixed. The level-channel title explicitly states that it is not an exact structural static-level test. The original test name remains in `original_test_name` within the source/export metadata; no restriction matrix or p-value is altered. Static two-channel and DJO1 aliases, plus one-coefficient benchmark/DJO0 sum aliases, are also labelled so repeated algebraic restrictions are not presented as independent evidence.
