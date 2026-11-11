# Reproducing the Singapore heat/economy project

This handoff has three modes: inspect the frozen HTML, verify the saved results, or refit the analyses from frozen inputs. The release manifest and verification receipts identify the exact delivered files. A successful check establishes its stated numerical, data or presentation scope; it does not establish a causal heat effect or clear every model's stationarity concerns.

## Obtain the frozen release

Repository: [calvintancy/heat-econ-singapore-replication](https://github.com/calvintancy/heat-econ-singapore-replication) (private). Release: [v1.0.0](https://github.com/calvintancy/heat-econ-singapore-replication/releases/tag/v1.0.0). Access requires permission to that private repository; no GitHub credentials are included in the project.

The release provides:

| Asset | Contents |
|---|---|
| `replication-inputs.tar.gz` | Cached official/derived inputs, processed data and immutable reference snapshots |
| `replication-outputs.tar.gz` | Saved results, exact samples, covariance matrices, figures, HTML and verification/provenance receipts |
| `replication-full.zip` | Portable source, inputs and outputs in one directory tree, without nested copies of the tar archives |
| `replication-manifest.json` | Source/member paths, sizes and SHA-256 hashes, plus archive hashes |

For a code checkout, restore the two data/result archives using the standard-library bootstrap:

```bash
gh repo clone calvintancy/heat-econ-singapore-replication -- --branch v1.0.0
cd heat-econ-singapore-replication
python3 scripts/restore_replication.py --repo calvintancy/heat-econ-singapore-replication --tag v1.0.0
```

The checkout is pinned to the release tag so later source edits are not mixed with the frozen assets. The bootstrap uses the existing authenticated GitHub CLI session. If the assets were downloaded separately, use their directory instead:

```bash
python3 scripts/restore_replication.py --archive-dir /path/to/downloaded/assets
```

Alternatively, extract `replication-full.zip` without flattening its directories. It already contains the source, `data/`, `outputs/` and `replication/manifest.json`; no Git clone or archive restore is needed. From that extracted project root, check its installed files with:

```bash
python3 scripts/restore_replication.py --verify-only
```

`--verify-only` checks the files already on disk. `--verify-archives --archive-dir /path/to/assets` checks the input/output tar archives and their members. The ZIP's embedded manifest omits its own ZIP hash to avoid a self-reference; the detached release manifest additionally records the complete ZIP hash. Restore verifies source compatibility and archive/member hashes before installing inputs/results; it does not overwrite source code.

## 1. Inspect the saved analysis

Open these files in a modern browser:

- `outputs/heat_economy.html`: the complete tabbed VA/IIP, annual, control, heterogeneity, diagnostic and paired-COVID report.
- `outputs/visitor_arrivals.html`: the separate visitor-arrivals report.
- `outputs/clustered_sector_sensitivity.html`: requested sector/industry-CR1 interaction sensitivity beside the original calendar-DK uncertainty.

The reports embed their analysis data, figures and browser code and can be inspected offline. Links to external publications still require internet access. The visitor analysis is separate: it is not a tab in the economic report and is not called by `scripts/run_all.py`. The replication wrapper can run both families sequentially without changing that separation.

No Python, Node or R is needed merely to inspect the saved HTML. The large files can take time to open; keep their original filenames and use a desktop browser.

## Environment for verification or refitting

Use **Python 3.11**, **uv 0.12.3**, and **Node.js 24**. Install the Python packages from the committed `uv.lock`, rather than resolving the dependency ranges again:

```bash
uv --version
node --version
uv python install 3.11
uv sync --locked --python 3.11
uv run python scripts/replicate.py inventory
```

The Python installation step can be omitted if Python 3.11 is already installed. Environment setup and the initial release download can require internet access. The analytical wrapper blocks Python HTTP requests during verification/refitting, so missing caches fail rather than silently fetching a newer statistical vintage.

Node runs the browser-JavaScript verification harnesses. The visitor validator uses `import.meta.dirname`; use Node 24 as specified. It requires only Node's built-in modules, with no npm install. Inspecting the delivered HTML and verifying existing results do not require R. No database or prior electricity-project directory is required.

Visitor verification refreshes the numerical audit receipts, then assembles the HTML from the saved native `gt` tables before checking its controls and embedded evidence downloads. This keeps timestamped receipts current in a moved checkout without refitting models or requiring R for verification.

Full main-analysis refitting additionally requires Poppler's `pdftotext` executable for the official IIP-method/source reconciliation. For example, install `poppler` with Homebrew on macOS or `poppler-utils` with the system package manager on Debian/Ubuntu. This is a real runtime dependency even though the PDFs are cached.

Full visitor-report regeneration additionally requires **R 4.6.1** and the locked `gt`/`jsonlite` dependency environment in `replication/renv.lock`. Set it up before starting the offline analytical wrapper:

```bash
Rscript --version
Rscript --vanilla scripts/setup_visitor_r.R .
Rscript --vanilla scripts/setup_visitor_r.R . --check-only
```

The helper restores `gt` 1.4.0, `jsonlite` 2.0.0 and their locked dependency closure, bootstrapping pinned `renv` 1.3.0. It installs into `outputs/visitor_arrivals/R-library/` by default; `--library PATH` selects another local target. Setup uses HTTPS/CRAN and can require a compiler. On Debian/Ubuntu, source builds can need `build-essential`, `pkg-config`, `libcurl4-openssl-dev`, `libxml2-dev`, `libnode-dev`, `libuv1-dev` and `libssl-dev`, in addition to the specified R version. The helper does not change a global R library or initialize a project `.Rprofile`. Its explicit R-version-mismatch override is outside the canonical reproduction setup. Native `gt::as_raw_html` rendering does not require Pandoc, TeX or a browser-rendering installation.

R renders the native visitor regression tables. Installed R libraries and compiled/native files are excluded from the archives and must be installed for the target platform. A main-only refit does not need R; a visitor-only refit does not need the main IIP PDF extraction step. Dependency downloads belong to setup, not to the analytical refit.

A fresh, separate R library was restored and checked with all 56 locked package versions and a native `gt` smoke render on R 4.6.1, `aarch64-apple-darwin25.4.0`. That test did not install globally or create a project `.Rprofile`. The Linux system-dependency list is preparation guidance; it is not a claim that the complete project has been runtime-tested on Linux.

## 2. Verify the existing results

The default is exhaustive verification:

```bash
uv run python scripts/replicate.py verify --all
```

A shorter smoke check is available:

```bash
uv run python scripts/replicate.py verify --quick
```

Quick mode checks representative main numerical analyses and all selected source-backed presentation checks. The visitor numerical verifier is monolithic, so a quick run that includes visitors still executes its full numerical audit and Node report test. Quick mode must not be described as an exhaustive numerical check of every main model.

Select a family when needed:

```bash
uv run python scripts/replicate.py verify --all --only main
uv run python scripts/replicate.py verify --all --only visitors
uv run python scripts/replicate.py verify --all --dry-run
```

These commands read the delivered estimates and independently check their definitions, samples, coefficients, uncertainty and presentation. They do not refit production models. They do write fresh verification receipts. The wrapper preserves/restores the two source method notes whose generated audit blocks are maintained by low-level verifiers, while retaining new proof JSON/CSV. Preserve an untouched extracted copy if you need the original release's output bytes as well as a working verification directory.

`restore_replication.py --verify-only` is a frozen-file integrity check, not a substitute for the numerical verifiers. After working results/receipts have been regenerated, their hashes may differ from the published output snapshot; inspect the new verification receipts rather than assuming a refit is byte-identical to the release.

## 3. Refit from the frozen inputs

Use a separate working copy with enough space for the release archives, extracted inputs/results and regenerated reports. Then run:

```bash
uv run python scripts/replicate.py reproduce --all
```

The wrapper runs the main VA/IIP/annual/COVID pipeline, regenerates the cached-calendar-derived visitor control, runs the visitor OLS/PPML, crisis and origin-cluster analyses, renders its native R/gt tables, builds its report, and performs the selected independent checks. Python programs run in separate subprocesses to isolate estimator ledgers. Inspect the exact plan without fitting with:

```bash
uv run python scripts/replicate.py reproduce --all --dry-run
```

`--only main` and `--only visitors` also apply to `reproduce`. A main-only run omits the visitor stage. A visitor-only run requires the restored main report and shared weather/control inputs already present in the release tree.

This is frozen-vintage replication. Do not use the low-level `run_all.py --refresh` option, delete caches to force redownloads, regenerate old reference snapshots or replace their manifests when reproducing this release. The paired comparison's explicit pre-fit checkpoint refresh is different: it records the freshly rebuilt parent results before paired fits and is part of its normal audit. It does not refresh raw sources or overwrite the immutable pre-Pyfixest/pre-NSA reference inputs.

## Files required beyond the main scripts

Keep the complete packaged directory structure. In particular, these are required rather than optional scratch files:

- `pyproject.toml`, `uv.lock`, the runtime version pins, `templates/`, `notes/`, and all project modules in `scripts/`.
- `data/raw/visitor_arrivals/calendar/extract_lunar_new_year.py`. This is original executable source stored beneath the data tree; it is retained as tracked source and in the full ZIP, not as a member of the input tar archive.
- Both `data/reference_before_pyfixest/` and `data/reference_before_nsa_inputs/`.
- The frozen migration input manifest at `outputs/pyfixest/migration_input_reference.json`, the paired audit's `outputs/covid_comparison/original_inputs_reference.json`, native estimator receipts and exact saved samples.
- The full source manifests and their matching cached payloads. A merged TableBuilder payload, its pages and metadata can all be needed for source-integrity checks.

Historical provenance notes/manifests can contain paths from the earlier electricity study's workstation. They identify provenance, not locations to mount on the reproducing machine. Runtime paths resolve within the project root; the frozen wind cache replaces the earlier EMA-directory dependency.

## What the receipts establish

The main source-analysis receipt currently records 1,510 native Pyfixest fits; annual, paired-COVID and sector-clustered interaction-sensitivity fits have separate ledgers. Counts are reconciled against model/outcome/unit inventories and should be read from each current manifest, not treated as an eternal hardcoded target. OLS and PPML visitor receipts likewise remain separate from the economic ledgers.

Sector-interaction models use their parent model's exact common sample. Eligibility follows available growth and controls, stated 90% coverage rules, matched start dates and the explicit COVID variants. There is no outlier- or significance-based sample trimming. Outcomes and exposure lags are constructed before endpoint exclusions; sample changes do not compress missing calendar dates or replace a YoY denominator.

Common-slope panels use unit/sector CR1, which permits dependence over dates within each unit; it does not protect against arbitrary cross-sector shocks, and there are few sectors. An individual industry's temperature slope is supported by that industry, so ordinary sector CR1 is not a reliable uncertainty calculation for it. Shared nuisance parameters can produce a small, nonzero clustered SE; it is not necessarily exactly zero. Sector-specific profiles instead use joint calendar DK/HAC, aggregating scores by actual date and using time-based reference degrees of freedom under the stated assumptions. Short time windows remain fragile.

The requested sector-clustered sensitivity is saved separately under `outputs/clustered_sector_sensitivity/`. It refits H1 on the exact parent common samples for broad/detailed VA and monthly/quarterly IIP, retains the calendar-DK results, and adds sector CR1 with `t(S−1)`. The main 1,510-fit receipt is not replaced. Coefficient/component covariance and HC0-scaled degeneracy diagnostics make the limitation visible: collapsed component or equality uncertainty does not receive an artificial p-value. Inspect this report as a sensitivity, not as validation that clustering a unit-specific slope by that same unit is reliable. Its independent verifier checks the full design, covariance propagation, exact samples and preserved DK comparisons.

Independent checks cover numerical reconstruction and the stated source/sample/presentation relationships. They do not reproduce every cited paper's empirical results. The temperature-horizon literature review proposes additional dynamic specifications; it does not mean those proposed models were estimated in this release.

Annual full controls have nine eligible years and only two effective time-series residual degrees of freedom. Their six-year COVID-excluded counterparts are underidentified and intentionally displayed as unavailable. Other short PM2.5-controlled windows and few-cluster inference remain fragile. Gapped exclusions are not compressed for ADF/KPSS; full-history diagnostics do not clear every shorter model window. Significant coefficients and successful numerical audits are not causal or stationarity certificates.

The source/data licences and redistribution boundaries are described in [DATA_SOURCES.md](DATA_SOURCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The completed separate-checkout full rebuild, tested environment and execution log are recorded in [the handoff validation](replication/VALIDATION.md).
