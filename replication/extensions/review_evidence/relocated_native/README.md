# Relocated native execution: compact evidence

This is a historical execution bundle, not a replacement result reference or a new source seal. Absolute run IDs, paths and timestamps are retained exactly. The original v1.0.0 references remain unchanged.

The dependency-corrected, guard-revised replay passed with **11,427 saved native fits**: 11,108 heat OLS, 299 arrivals OLS and 20 arrivals PPML. It used an existing locked Python environment. The initial relocated copy omitted `.git`, `.venv` and the pre-existing `dist` tree; producers could create their own scratch directories. It does **not** establish a fresh-environment rebuild or final sealed-source reproduction. HTTP was blocked and native work was sequential with one-thread limits.

All ten mathematical verifier scripts are covered. Four completed, unaffected audits were reused: crisis, annual crisis, complete VA controls and annual literature. The remaining six scripts were rerun after the explicit guard-only source boundary: subannual literature, seasonal literature, full-controls literature, monthly arrivals, annual/PPML companions and origin-year arrivals. These cover eleven numerical family receipts. Counts describe the saved execution records, not independent hypotheses or weather realizations.

The first dependency-order failure, external counter error and subsequent derived-proof lineage failure remain failed historical receipts. `resumed_production_verification.json` reconciles the counter: rebuilt deterministic ledger CSVs need not change SHA. The final successful `guard_revision_run.json` records the authorized source transition and reused receipts; it does not silently turn earlier failures into successes. The separate guard AST/diff evidence is in the parent `derived_proof_guard_revision.json`, with its executable check in `scripts/verify_derived_proof_guards.py`.

Native coefficients, residuals and covariance arrays were recomputed and independently checked; this evidence is computational, not a causal, stationarity or nominal-test-calibration certificate. The actual archive does not include the duplicated COW data/model tree or binary installations. Use the current project driver for reproduction, not the archived driver with historical absolute paths.

Inventory SHA-256: `1f42f5d9f94da1af27403e6a6820d1ca02dbb9381fecf2dd6a5e9b6df63c2667`. `SHA256SUMS` also covers this README and the inventory, excluding only its own digest. On macOS use `shasum -a 256 -c SHA256SUMS`; on Linux use `sha256sum -c SHA256SUMS`, from this directory.

Receipt/log contents are byte-exact. `.log` transcripts were copied to `stdout/*.txt` with no content edits; `inventory.json` maps each portable member to its historical source path. All selected text passed the package private-key/token/numeric-license guard before copying. This is a targeted scan, not an exhaustive privacy certificate.

| Member | Bytes | SHA-256 |
|---|---:|---|
| `derived_verification_lineage_failure.json` | 289 | `a12b98d071861f8d710b3ad57bde60855c9ad7efad8f84908b0b2d6011d5196a` |
| `environment.json` | 1,727 | `4bf0f7e43a8ffd4038a56773e85ea2b7b51ee015915d934f1eb41e3078e088a7` |
| `execution_manifest.json` | 890 | `5311c6081ee79e66ca27afeca721b23c065ec1a87e2b3f4c7e8233bd2049ab20` |
| `execution_sources/guard_revision_execution_driver.py.txt` | 5,770 | `b17f5b07c4fa2a52acaf8a5c8eebb2c5e9fadd5f0d4e35f81b6c24ffd7b7b094` |
| `guard_revision_run.json` | 37,672 | `a2700901744ec96ebb8bc382120564c5e462931da531f86ba4fa728c8b06ff69` |
| `numerical_receipts/complete_va_controls.json` | 10,797 | `634c71190c03b8b6135eb7a4c6e29a4f027b2459494e8a78ec50eb8723f70ca1` |
| `numerical_receipts/crisis_sensitivity.json` | 27,083 | `afb5083b64663c3342d21766ca46fb7589a23ac9aa7ce344483195e38613620c` |
| `numerical_receipts/crisis_sensitivity__annual.json` | 7,411 | `b0f11b448ba9ef7a904b5e399a3601a428a4d3c7963f33f52d08e94d5fc896b9` |
| `numerical_receipts/literature_sensitivity.json` | 20,986 | `1efeb7c2badbff90d26b11ce841d941b75ecb89e622801b96bdecb2b00f8d2f2` |
| `numerical_receipts/literature_sensitivity__annual.json` | 3,642 | `ce58cc94b27cc996fc6e69ede364fc63b7e37fdc0b61bb6e1026f08d4309f5c8` |
| `numerical_receipts/literature_sensitivity__annual_seasonal.json` | 257,674 | `667e30b59fb311bd2f5a49fc31dcd88290d0b07120104a841d0bf7cce9085bae` |
| `numerical_receipts/literature_sensitivity__full_controls.json` | 21,264 | `f733f781848f17c9b9b05edd4a8112c60b16fa1b6a21b860f4209fbdb734a73b` |
| `numerical_receipts/visitor_rigour__annual.json` | 23,398 | `f20809abcc2845caef0ba2bdbf191d1afe955391df68ec6bd1bb1c08d3a44a11` |
| `numerical_receipts/visitor_rigour__monthly.json` | 79,146 | `c3868edba705c999d648e71dec45cbcabbd0a798a2733ffe8d8011b7f8a8bce7` |
| `numerical_receipts/visitor_rigour__origin_year.json` | 29,277 | `55f905ceae38bc8758b41b3aa9372408c287914c8ea4036a1c3484509ad2e37d` |
| `numerical_receipts/visitor_rigour__ppml.json` | 28,311 | `69bfe7d84032ca4e012cb729d761a7c88d2f2e658e9dcef6210c756412cb4bf6` |
| `numerical_run.json` | 2,628 | `373706da094bdf9a579401664001b38a2ed2f344b7a1964d22651b03771e94e8` |
| `resumed_production_verification.json` | 35,206 | `00447c5960407dacfef318115a17923d860b3618a5be56ea4d41bf185f991e26` |
| `resumed_run.json` | 31,709 | `2a86cb89286025d3c009760d5cfb9671d6e50aea1885c30090ca0999d91ba5ed` |
| `run.json` | 33,798 | `62e8b784d6824b3ed5dd9c9198afe65e95952b0e1baf3a0ecbc9a62cf07de3bf` |
| `stdout/01-crisis_sensitivity.py.txt` | 5,151 | `d4a3d2e64ffaad9eabc91034fc98c724fd5610e63146b487370ab463b7ca0c07` |
| `stdout/02-annual_crisis_sensitivity.py.txt` | 7,837 | `ac0ef7ceef3da07628e189e04037e6435ac53b47eb0ba226a8db2b1e6f85d735` |
| `stdout/03-complete_va_controls.py.txt` | 805 | `8015862761eae9afb97500be2fb605da36bcc1c0c5d6aeeb106867ed43049078` |
| `stdout/04-annual_literature_sensitivity.py.txt` | 52,574 | `7021aaa85d59966cfbf94f0df526329134491c7b791a13105674ad28f7700eef` |
| `stdout/05-literature_sensitivity.py.txt` | 15,457 | `b4bf3cbace1be1097734f26029d1a5e0e98ebb34293c4948884c2c824338c06f` |
| `stdout/06-annual_seasonal_literature.py.txt` | 1,675 | `0bbbd2027d6dbb284dd9675d5379a3276d334e978e82da7eb12fe32ad20f7e62` |
| `stdout/07-literature_full_controls.py.txt` | 45,439 | `666e3ebdcf9c17ad70a7b5a406687d5572528f45763a4c0e62a800b98faf5d29` |
| `stdout/08-literature_rank_correction.py.txt` | 185 | `2b31669260c2694b7559b17b6f338b5c93848d3c61fe161f7e618c95b239325b` |
| `stdout/09-literature_restrictions.py.txt` | 85 | `ffdfdf3e86bdd76f5415e5805dfd3b38b23190fb2c4526f1d524fe850286a9e2` |
| `stdout/10-literature_full_controls_restrictions.py.txt` | 963 | `b38221e8d7799c7c20bc8c68479393efc1c6e6d42e12f721b680ef80a7dc902f` |
| `stdout/guard-v2-01-literature_sensitivity.py.txt` | 15,457 | `b4bf3cbace1be1097734f26029d1a5e0e98ebb34293c4948884c2c824338c06f` |
| `stdout/guard-v2-02-annual_seasonal_literature.py.txt` | 1,675 | `0bbbd2027d6dbb284dd9675d5379a3276d334e978e82da7eb12fe32ad20f7e62` |
| `stdout/guard-v2-03-literature_rank_correction.py.txt` | 185 | `2b31669260c2694b7559b17b6f338b5c93848d3c61fe161f7e618c95b239325b` |
| `stdout/guard-v2-04-literature_restrictions.py.txt` | 85 | `ffdfdf3e86bdd76f5415e5805dfd3b38b23190fb2c4526f1d524fe850286a9e2` |
| `stdout/guard-v2-05-literature_full_controls.py.txt` | 45,439 | `666e3ebdcf9c17ad70a7b5a406687d5572528f45763a4c0e62a800b98faf5d29` |
| `stdout/guard-v2-06-literature_full_controls_restrictions.py.txt` | 85 | `d8f9fd006fa2918971c7271f6d92165742bc8d4a25538fc82e26a6766b07a0f0` |
| `stdout/guard-v2-07-visitor_rigour_monthly.py.txt` | 39,250 | `620c40e61093e5689ce4b0a1072e28a9e17c4a2f9c253cf63c4fc3dc2a773b76` |
| `stdout/guard-v2-08-visitor_rigour_companions.py.txt` | 1,465 | `5f08add319407943c602a4bddc4d27e4ee89aeefb9a390da9423237d61bef66a` |
| `stdout/guard-v2-09-visitor_origin_year.py.txt` | 3,308 | `e7031aeff2a4a6eb61ba9e3dc8cd5715963e6fdabbd5c9dcb6a5937c5d4d97d3` |
| `stdout/guard-v2-10-verify_literature_sensitivity.py.txt` | 2,159 | `424fd25bec9f7b30b7df9ed54fe274b6856540588c45af2da0cdb36169f24af4` |
| `stdout/guard-v2-11-verify_annual_seasonal_literature.py.txt` | 113 | `e3b2f938f73e5fa69a249266f173cbcce9d004601e11c429d314ef7a28b0b61c` |
| `stdout/guard-v2-12-verify_literature_full_controls.py.txt` | 113,783 | `410a4a891cd0f309d8025fd752654bf51796de23ad2f0eca0383d72173d698ec` |
| `stdout/guard-v2-13-verify_visitor_rigour.py.txt` | 904 | `140e2aca7a4e9421d67752f5e65288c37df1557cdcd341b789dbf0063f172e6c` |
| `stdout/guard-v2-14-verify_visitor_rigour_companions.py.txt` | 94 | `cef0e6c92cb896cc88d5d908ed528009e0be592822d5dc3b00ff2d72191502da` |
| `stdout/guard-v2-15-verify_visitor_origin_year.py.txt` | 1,037 | `d4e8dadd3b6b48840ed6aa42101e184a66df0b7927edb80b126d9d2a5a498658` |
| `stdout/resume-01-literature_full_controls.py.txt` | 45,439 | `666e3ebdcf9c17ad70a7b5a406687d5572528f45763a4c0e62a800b98faf5d29` |
| `stdout/resume-02-literature_full_controls_restrictions.py.txt` | 85 | `d8f9fd006fa2918971c7271f6d92165742bc8d4a25538fc82e26a6766b07a0f0` |
| `stdout/resume-03-visitor_rigour_monthly.py.txt` | 39,250 | `620c40e61093e5689ce4b0a1072e28a9e17c4a2f9c253cf63c4fc3dc2a773b76` |
| `stdout/resume-04-visitor_rigour_companions.py.txt` | 1,465 | `5f08add319407943c602a4bddc4d27e4ee89aeefb9a390da9423237d61bef66a` |
| `stdout/resume-05-visitor_origin_year.py.txt` | 3,308 | `e7031aeff2a4a6eb61ba9e3dc8cd5715963e6fdabbd5c9dcb6a5937c5d4d97d3` |
| `stdout/verify-01-verify_crisis_sensitivity.py.txt` | 1,102 | `ef9ebaef86d88f0d71464cbcf1f6078ab7424c4063b710f13c9c62514eb2262c` |
| `stdout/verify-02-verify_annual_crisis_sensitivity.py.txt` | 479 | `cf87fa25714735b690020c9db5f7e30565b98c25b54f9f6478105444a2e75fb0` |
| `stdout/verify-03-verify_complete_va_controls.py.txt` | 101 | `1a003e930c3373111d4266c11be582f5817d82f68f6094bba14f43716f4af436` |
| `stdout/verify-04-verify_annual_literature_sensitivity.py.txt` | 121 | `56c70e8096a3035c1ada3470568165a9318db451e82fd475206d28d8da54aabd` |
| `stdout/verify-05-verify_literature_sensitivity.py.txt` | 1,605 | `af3f57b71dd35a4a135bc9da9b2b6c1fd4e4cf1fa4e23a4668e269bb9781832f` |
