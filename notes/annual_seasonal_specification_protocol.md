# Annual calendar-quarter exposure companion

Supplement to the frozen literature specification protocol, 11 November 2026.
This is a post-exploration robustness exercise, not a preregistration.

Colacito, Hoffmann and Phan, Richmond WP18-09 (methods version of their 2019
JMCB paper), equations (4) and (8), motivates annual output growth on seasonal
temperatures, lagged annual growth and current/prior-year seasonal exposure.
[Primary working paper](https://www.richmondfed.org/-/media/RichmondFedOrg/publications/research/working_papers/2018/pdf/wp18-09.pdf).
The inspected working paper uses nominal GSP in its baseline, Fahrenheit,
deseasonalized spatial weather, state/year FE and output-share weights. These
features are not claimed reproduced by our Singapore real-output adaptation.

Use the same fourteen annual contexts, including complete GVA12/20 and published
national totals on original and matched histories. Retain NSA annual real-output
log growth as primary, SA as comparison. Q1–Q4 mean air temperature enters as four
separate exposures; a second model includes the same four quarters from the
previous year. These are Singapore calendar quarters, not winter/summer labels.
Matching quarter-specific rainfall totals and current annual humidity are added
controls. Both specifications have an additional AR1 annual-growth sensitivity.
Sector FE and a common linear trend replace the infeasible full annual year FE.
H1 interacts every temperature feature with sector, sharing control and AR slopes.

Weather histories use the full cached calendar, before output-start truncation.
Lagged growth uses exact previous calendar years from the complete raw annual
output-growth registry, before weather or comparison-panel truncation. Show
model-own histories and a common maximum-AR1-history calendar across the four
models. Current-date crisis exclusion and strict output-support exclusion use
the same fixed broad policy as the other extensions. The strict matched family
requires t−2 through t to be clean because lagged annual growth itself compares
the two previous annual output levels. Non-AR models share that strict comparison
calendar. A model-own-history option must still state its exclusion rule clearly.
For explicitly matched contexts, intersect eligibility with the target context's
model/history support. Distinguish this comparison-date alignment from genuinely
missing raw history; an early year is not called missing when its raw value exists.

Save all current/prior-quarter coefficients, within-quarter current-plus-lag sums,
their complete covariance and relevant equality/zero-response restrictions.
Sector effects are reference-invariant totals. A coefficient sum in an AR model
is a direct conditional response, not an unconditional long-run multiplier.
Include native designs/receipts, exact sample masks, mean/SD Y, residual time df,
and explicit unavailable equations/tests. Do not silently remove weather terms,
controls or trends to make a short record estimable. Use the same CR1/common
panel and calendar-HAC/profile inference distinction as the other extensions.

This companion implements seasonal timing as a robustness question. It does not
identify a permanent climate effect, establish stationarity, correct dynamic-FE
bias, or eliminate omitted Singapore-wide shocks.
