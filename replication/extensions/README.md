# Heat, production and arrivals: audit extension

This is a separately versioned local extension of the published v1.0.0 replication
package. The original release tag, asset hashes, input/result references and
benchmark tolerances are unchanged. No updated GitHub release is implied.

From the project root, with the frozen inputs and original outputs restored:

```sh
uv sync --locked --python 3.11
uv run --locked python scripts/replicate_extensions.py inventory
uv run --locked python scripts/replicate_extensions.py verify
uv run --locked python scripts/replicate_extensions.py reproduce
```

`verify` independently checks all extension calculations and rebuilds/checks the
HTML, without refitting economic models. It refreshes heat and arrivals diagnostics after
the current parent verification receipts. `reproduce` first runs seven native heat
producers, three saved-covariance postprocessors and three arrivals producers,
then the independent numerical and presentation checks. The extension records
11,108 heat OLS, 299 arrivals OLS and 20 arrivals PPML fits. The original 4,781 OLS
and 12 PPML records remain separate. Counts describe execution, not independent
hypotheses or weather realizations.
Both use cached data with HTTP blocked and check the original frozen input-member
hashes before and after execution. A separate `source_manifest.json` binds this
extension's source. Changing code requires an explicit new source revision;
neither command silently reseals a manifest or relaxes tolerances.
All 526 original input members remain frozen. The extra official HKO 2007 lunar
calendar and its source/hash are versioned under `supplementary_inputs/calendar/`;
no original calendar file is replaced. Use `--dry-run` to inspect ordered steps.
The original `Frozen replication` GitHub workflow targets the v1.0.0 package;
run it at that tag. This extension uses the separate driver above.

A relocated native rebuild exposed an ordering defect and derived-proof hashes
incorrectly treated as model inputs. Rank/span postprocessing now precedes fitting
its full-controls child. Three narrow guard revisions permit replacement
verification/presentation records while protecting the actual analytical artifacts.
`verify_derived_proof_guards.py` checks archived source preimages, unchanged model
ASTs and actual tamper boundaries. The arrivals review-equivalence check separately
compares the final refit with its explicitly archived reviewed snapshot; it does
not relabel a historical external review as a review of later source changes.
That snapshot bridge requires exact array identity, distinct from the independent
auditors' stated numerical tolerances. Cross-platform differences require separate
adjudication; the baseline is never silently refreshed.

The model/effect NPZ files and CSVs must be present for existing-output verification.
The master result is `outputs/heat_economy.html`; standalone new reports are
`outputs/crisis_sensitivity.html`, `outputs/literature_sensitivity.html` and
`outputs/aea_audit.html`, `outputs/extension_stationarity.html`,
`outputs/visitor_arrivals_rigour.html`, `outputs/stata_key_results.html` and
`outputs/clustered_sector_sensitivity_qualified.html`. The arrivals chapter embeds
its diagnostics and native Stata tables. The exact-calendar
unit-root appendix maps all new requested equations to their actual outcome and
stochastic-regressor arrays. Short and gapped samples remain untestable; ordinary
ADF/KPSS screens do not establish universal stationarity. The literature guide states model/version differences,
exact outcomes and controls, FE, sample losses, inference, unavailable equations
and interpretation boundaries. Its numerical checks do not establish causality,
stationarity, valid small-sample test size or AEA conference acceptance.

The audit reproduces a known-null calibration failure for a selected large joint
calendar-Wald test. Both driver modes also replay selected arrivals Gaussian-null
diagnostics; these are reference-law checks, not the empirical arrivals error law
or a replacement p-value rule. Full rank does not establish valid test size; nominal p-values
and within-table Holm adjustments do not control selection over the entire grid.
Individual sector-CR1 interaction inference remains unsupported. The qualified
viewer suppresses its p-values, intervals and stars, retaining estimates and labelled
historical downloads. Growth models with short or gapped exact calendars remain
untestable by ordinary ADF/KPSS; compatible long histories do not clear other windows.
The arrivals appendix does not establish stationary growth or impose ordinary I(0)
tests as a prerequisite for a Poisson conditional-mean model.

Each driver execution writes an authoritative running/passed/failed receipt and
per-step logs under `replication/logs/extensions/`. A failed subprocess cannot be
overridden by an older file that happens to say passed. Original verify-only
bookkeeping separately archives fresh receipts under
`replication/logs/verification-runs/`, retaining the eight historical checkpoint
JSON records without confusing them with current evidence.

The initial parent-limited subannual lag run is superseded audit evidence. The
final history rule builds growth lags from the full raw output calendar, then
applies target-calendar alignment and fixed crisis masks. See
`notes/literature_history_alignment_v2.md` and the report for that correction.

## Actual Stata MCP implementation

Licensed Stata 17/IC with `reghdfe` 6.13.1 completed 99 heat and 38 arrivals keys
through real MCP stdio. Native logs, reusable do/Mata files, e(sample), full selected
covariances, native esttab tables and pinned MCP/environment evidence are under
`outputs/stata_key_results/`. Separate Python ports independently validate the native
outputs. The default extension driver verifies those stored runs; it does not
substitute Python for a fresh licensed Stata execution. Its native/project
finite-sample conventions remain distinct; singular native interaction-cluster
inference stays unavailable. The selected checks do not cover every lag/nonlinear
or PPML model in Stata.

To rerun in a clone, restore inputs and regenerate clone-specific do paths, then
execute through MCP with licensed Stata and matching reghdfe/ftools/estout packages:

```sh
uv run --locked python scripts/stata_key_results.py
uv run --no-project --python 3.12 --with-requirements outputs/stata_key_results/mcp/requirements.txt python scripts/stata_mcp_client.py --receipt heat_key_results --timeout 1800 --do-file outputs/stata_key_results/do/key_results.do
uv run --locked python scripts/verify_stata_key_results.py

uv run --locked python scripts/stata_visitor_rigour.py
uv run --no-project --python 3.12 --with-requirements outputs/stata_key_results/arrivals/mcp/requirements.txt python scripts/stata_mcp_client.py --receipt arrival_key_results --receipt-dir outputs/stata_key_results/arrivals/mcp --timeout 1800 --do-file outputs/stata_key_results/arrivals/do/key_results.do
uv run --locked python scripts/verify_stata_visitor_rigour.py
```

Use `--stata-path` when the executable differs. Existing environment evidence is
historical; explicitly recapture a different native environment. Original-release
validation in `replication/VALIDATION.md` does not by itself validate this extension.

An optional real-browser smoke check runs after the source-backed checks:

```sh
uv run --locked --with playwright==1.65.0 python scripts/verify_browser_reports.py
```

It requires installed Google Chrome (`HEAT_CHROME_EXECUTABLE` overrides the macOS
default). It checks offline native tables, unavailable states, exact downloads,
master iframe navigation and mobile overflow. Receipts/screens under
`outputs/browser_verification/` bind the exact HTML generation tested; they are
presentation evidence, not numerical or scientific-validity certificates.

## Portable extension package

The extension uses several independent ZIP parts with nonoverlapping files, each
below 1.8 billion bytes. Download all parts plus `heat-econ-extension-manifest.json`
and `heat-econ-extension-SHA256SUMS`. Extract every part into the same empty folder;
then run the standard-library integrity check before executing analysis:

```sh
python3 scripts/package_extensions.py verify --root .
uv sync --locked --python 3.11
uv run --locked python scripts/replicate_extensions.py verify
```

Alternatively, from a source checkout, verify archive/member hashes while extracting:

```sh
python3 scripts/package_extensions.py extract --manifest ASSETS/heat-econ-extension-manifest.json --archive-dir ASSETS --root NEW_FOLDER
```

Keep ZIPs and detached transport metadata outside the extracted project. The
embedded inventory is under `outputs/extension_package/`; `.git` is unnecessary.
Archive checks certify file integrity. A separately recorded analytical execution
establishes only the actual verification or reproduction scope it completed.
For maintainers, create a new source manifest explicitly with
`scripts/seal_extension_sources.py`, complete verification, then use
`scripts/package_extensions.py build --source-manifest PATH --out-dir dist/extension-package`.
The build refuses missing completion evidence, modified original references,
unsealed source changes, credential/license identifiers and unsafe archive paths.
