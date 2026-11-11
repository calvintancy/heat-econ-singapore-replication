# Matched origin-by-year fixed-effect robustness

This targeted robustness family addresses origin-specific annual demand and trend
confounding raised by the external inference review. It is an exploratory addition,
not a preregistration, and is fixed across all declared cuts/outcomes/samples before
reading this family's estimates. It does not replace the shared-year main equation
or extend the lag-length grid.

Run exactly 24 native PyFixest H1 equations: nine regions / 42 named markets,
MoM / YoY NSA log growth, long / full current controls, and all eligible /
event-date exclusion / strict output-support exclusion. Reuse each corresponding
`own_current` sample from the verified monthly family without any date, unit,
positive-count, coverage or holiday-history change.

Replace common year FE by origin × year FE. Retain origin × calendar-month FE,
temperature main + origin interactions, and shared current rainfall, humidity,
Lunar New Year current/prior-month days, plus wind/PM2.5 in full controls. There
are no additional weather or growth lags. Reference-origin interaction remains a
zero normalization; report total temperature effects with their complete covariance.

Origin-year FE absorb annual conditions separately by origin, including arbitrary
origin-specific annual trends. They leave within-year temperature variation after
seasonality. They do not absorb every monthly origin-specific shock or create
cross-destination exposure variation. Full actual-month FE would instead absorb
the common temperature level and identify relative origin effects only; those
point identities receive separate bounded Stata checks.

Keep calendar Bartlett HAC12/24 and the declared maximum-local-rank correction,
with explicit approximate reference df. Native point rank and temporal inference
availability remain separate. Origin clustering does not supply reliable H1
absolute-slope inference. Joint Wald statistics are nominal asymptotic diagnostics,
without a claim of calibrated high-dimensional significance. Retain unavailable
requests and small-time-df cautions rather than dropping controls.

This changes deterministic nuisance restrictions only. Exact raw Y and stochastic
weather bases are identical to the matched own-current equations; diagnostic links
should therefore reuse the same array IDs while recording the additional model
coverage. Do not test FE indicators or FE-projected series as new unit-root inputs.

Independent verification must rebuild the enlarged FE space, reproduce native
coefficients/residuals/full covariance and main-plus-interaction effects, and prove
exact calendar/value equality with the parent samples. Comparisons concern the
same observations. They are not independent new weather realizations or a causal
certification.

Origin-year controls are also used in cross-country tourism-demand research, for
example the [global natural-disaster/tourism study](https://doi.org/10.1016/j.tourman.2020.104080).
That study's bilateral data and disaster treatment differ from this one-destination
temperature analysis. Here the transferable component is flexible origin-year
conditioning, with the remaining identification limits stated above.
