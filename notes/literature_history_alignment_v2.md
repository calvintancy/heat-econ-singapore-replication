# Corrected history boundary rule for literature extensions

Version 2, 11 November 2026. This supplements and supersedes the parent-only AR
history rule in `literature_specification_protocol.md` for the subannual dynamic
extension. The separate annual six-model extension contains no lagged dependent
variable, so its frozen numerical results are unaffected.

An independent boundary audit found that weather-complete and matched-date
parents can start later than their raw output series. Looking up lagged growth
only in those cropped parents can therefore discard an otherwise valid current
observation. This is an eligibility-construction defect, not evidence that the
raw observation is missing. The initial subannual parent-limited run is retained
as superseded audit evidence, not presented as the final specification.

Construct SA/NSA output growth and all its lags from the complete raw official
output calendar. Construct weather lags from the complete cached weather
calendar. Only then apply the current-outcome definition, required-history
eligibility, explicit comparison-date matching, and the fixed crisis masks.
Never bridge a gap by using the previous retained observation, and never impute
an unavailable raw output or growth rate.

Every current/DL/dynamic/two-channel model still shares the maximum required
growth-history eligibility within each context/outcome. For contexts explicitly
matched to another sector definition, intersect that eligibility with the target
context's corresponding maximum-history support. This preserves a true common
calendar comparison even when a national total has longer raw history than a
detailed partition. Original-history contexts retain all usable raw dates.

Report three distinct causes of sample loss: unavailable raw output/weather
history, an explicit target-calendar alignment, and the declared crisis mask.
Do not describe target alignment as missing source data. Strict crisis masks
still cover the complete output support: t−h−1 through t for one-period growth
with h growth lags, and t−2h through t for YoY growth with h growth lags. These
rules are based on data support and calendar policy, never observed growth size,
statistical significance or a desired regression result.

Re-estimation uses a fresh native PyFixest ledger and independent raw-calendar,
design, coefficient, covariance and restriction checks. No frozen v1.0.0 source,
input/result reference, benchmark tolerance or original analytical result is
changed to make the corrected extension pass.
