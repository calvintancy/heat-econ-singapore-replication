# Quarterly YoY growth and the temperature horizon

Methods review, 11 November 2026. This note supplements the earlier AEA review. It examines the specifications and relevant methods sections of primary papers, rather than treating the words “temperature shocks” or “growth” in an abstract as definitions. It does not change any regression, sample or preferred specification in the saved results.

## Assessment

Quarterly YoY output growth does **not automatically require** YoY temperature differences. However, the intuition has a precise foundation: when heat changes the current level of output temporarily, the temperature in the comparison quarter affects the denominator. A differenced output-level model therefore involves the difference between current temperature and temperature four quarters earlier.

A growth mechanism gives a different exposure window. If each quarter's growth responds to its temperature, YoY log growth accumulates those effects over four quarterly growth intervals. A model allowing both mechanisms should accommodate endpoint differences and the intervening temperature history.

Current-quarter temperature alone is a distinct conditional projection. The annual-growth literature is not sufficient justification for treating it as the uniquely preferred quarterly YoY specification. Temperature measured in levels, a deviation from normal seasonal temperature, a change from last year's temperature and a rolling-year exposure are different objects. A weather anomaly is not automatically a YoY difference.

## Closest empirical precedent

Nguyen and Pienknagura's **Rising Temperature, Nuanced Effects: Evidence from Seasonal and Sectoral Data** was published in *Environmental and Resource Economics* in 2025. The exact equations and lag lists inspected here come from the open **IMF WP24/202**, because the final article's main body is subscription-restricted. The publisher's visible notes independently confirm a preference for actual rather than seasonally adjusted output and for temperature levels. [Published article and notes 6–7](https://link.springer.com/article/10.1007/s10640-025-01034-5); [open working paper](https://www.imf.org/-/media/files/publications/wp/2024/english/wpiea2024202-print-pdf.pdf).

At horizon zero, working-paper equation (6), printed p.9, uses quarterly sectoral YoY VA growth and current quarterly temperature **with four temperature lags**. It also includes current/four-lag precipitation, four lagged YoY growth rates, country FE and actual date-quarter FE. Later local-projection horizons additionally condition on intervening weather. This supports a dynamic weather specification, not current temperature alone and not a compulsory temperature difference.

The theoretical equations (3)–(4), printed p.5, identify the difference-only model as a special case without temperature effects on productivity growth. The empirical text calls the outcome YoY growth; its precise computational log-versus-simple-percentage convention is not separately confirmed. Its exposure is a quarterly average of daily **maximum** temperatures, differing from our average-air-temperature series. These distinctions prevent a claim of replication.

## Other primary specifications inspected

| Paper/version | Outcome and weather specification | What it establishes for this question |
|---|---|---|
| Dell, Jones and Olken (2012), *AEJ: Macroeconomics*, pp.72–74, equations (3)–(4). [Published PDF](https://economics.mit.edu/sites/default/files/publications/120626%20Climate%20Published.pdf#page=8) | Annual per-capita output growth on annual temperature/precipitation levels, with distributed-lag variants. Our earlier replication-code inspection confirms the log-growth transformation. | Growth outcomes can use weather levels and lags; the level-only channel imposes cancelling lag responses. Annual output growth is not quarterly YoY growth. |
| Burke, Hsiang and Miguel (2015), *Nature*, supplementary section B.2, equation (15), printed pp.38–39. [Supplement](https://web.stanford.edu/~mburke/climate/BurkeHsiangMiguel2015_SI.pdf#page=8) | Annual real GDP-per-capita log growth on a quadratic in annual temperature, precipitation terms, country/year FE and country-specific trends; lag sensitivities are separate. | Another annual level-of-weather specification, not a rule about how to transform quarterly temperature. |
| Colacito, Hoffmann and Phan (2019), *JMCB*; methods inspected in Richmond WP18-09, equations (4), (8), printed pp.12,15. [Primary PDF](https://www.richmondfed.org/-/media/RichmondFedOrg/publications/research/working_papers/2018/pdf/wp18-09.pdf#page=14) | Annual GSP growth on current-year seasonal temperature means. The extension includes current and previous-year seasonal temperatures separately and tests their sum. | Seasonal exposure timing and lag restrictions matter. In this WP, nominal GSP is baseline and real GSP a sensitivity; log-versus-ratio growth is not explicitly confirmed in the inspected text. |
| Kalkuhl and Wenz (2020), *JEEM*, equation (6), p.4; variants p.5. [Published PDF](https://publications.pik-potsdam.de/pubman/item/item_24336_4/component/file_24424/24336oa.pdf#page=4) | Annual per-capita regional-output log growth, with weather levels, changes, interactions/quadratics and lag variants in a framework distinguishing level and growth effects. | Temperature changes and levels can enter together; these terms encode different mechanisms. Annual temperature is a mean and precipitation an annual total. |
| Newell, Prest and Sexton (2021), *JEEM* 108, 102445, equation (1) and restrictions, pp.7–8. [Published PDF](https://dukespace.lib.duke.edu/server/api/core/bitstreams/5b11618d-ad09-4b89-873e-8abbbd75435b/content#page=7) | Annual real GDP-per-capita log growth on weather functions at lags 0–4. Different level/growth models impose different coefficient restrictions. | Differencing is a model restriction, not an outcome-label convention. For nonlinear effects, differencing temperature squared differs from squaring a temperature difference. The study highlights model uncertainty. |
| Hoang, Le, Nguyen and Spatafora (2024), IMF WP24/178, equation (4), printed p.10. [Primary PDF](https://www.imf.org/-/media/files/publications/wp/2024/english/wpiea2024178-print-pdf.pdf) | Monthly YoY **log nighttime lights**, using current temperature and temperature twelve months earlier separately, precipitation indicators, date FE and cell × calendar-month FE. | A direct subannual YoY example that retains both endpoints without forcing equal and opposite coefficients. Nighttime lights are an activity proxy, not measured VA; authors overlap with the quarterly paper. |
| Kahn et al. (2021), *Energy Economics* 104, 105624, equation (1), p.4 and Table 2, p.6. [Published PDF](https://academicweb.nd.edu/~nmark/Climate/Pesaran_EnergyEconomics.pdf#page=4) | Annual real GDP-per-capita log growth in a dynamic model. Preferred symmetric weather terms are changes in the absolute value of a scaled deviation from a preceding 30-year moving temperature norm, with lags. | An example using transformed differences, but neither raw temperature changes nor fixed-climatology anomalies. Its preferred term must not be relabelled as ordinary YoY temperature. |

These papers are examples of model choices, not independent votes for one Singapore coefficient. They differ in output measure, country coverage, temperature construction, controls, fixed effects and estimand. Their reported results were not replicated in this review.

## Why the proposed mechanism changes the answer

The following is our algebra, motivated by the frameworks above. Write `z(t) = 100 ln quarterly output(t)`, so the units are log points. Hold other terms explicit rather than assuming they disappear.

**Temporary output-level channel:**

```text
z(t) = baseline(t) + b × T(t)

YoY growth(t) = baseline(t) − baseline(t−4)
              + b × [T(t) − T(t−4)]
```

If both endpoint quarters are equally unusually hot, heat can depress both output levels while contributing zero to their YoY growth comparison. Using only current heat would omit that comparison-quarter channel unless the omitted exposure is otherwise adequately accounted for.

**Quarterly growth channel:**

```text
z(t) − z(t−1) = baseline growth(t) + c × T(t)

YoY growth(t) = sum of the last four baseline quarterly growth terms
              + c × [T(t) + T(t−1) + T(t−2) + T(t−3)]
```

The four-quarter mean represents the same restricted exposure as the sum, with its regression coefficient four times the per-quarter coefficient. This implication assumes equal contemporaneous quarterly growth responses, not arbitrary dynamic effects.

**Both channels:**

```text
YoY growth(t) = other terms
              + b × [T(t) − T(t−4)]
              + c × sum(T(t), T(t−1), T(t−2), T(t−3))
```

A fully unrestricted five-term temperature model can accommodate both and relax their timing assumptions. In a model with coefficients on `T(t),...,T(t−4)`, the contemporaneous level-only hypothesis jointly requires the three intermediate coefficients to be zero **and** the two endpoint coefficients to sum to zero. A zero sum alone is a more general restriction and does not establish this precise static model.

To call a regression the exact differenced counterpart of a level model, transform the weather controls and deterministic terms consistently too. Reusing fresh sector-season/year FE after changing temperature does not by itself establish exact algebraic equivalence. For nonlinear exposure, difference the basis functions: for example, `T(t)^2 − T(t−4)^2`.

Hsiang's methodological review discusses conditioning on exposure histories and temporal displacement using distributed lags (2016, section 4.2.1, equation (33), p.64). [Published review PDF](https://static1.squarespace.com/static/55667009e4b04bbb290cc837/t/5b6b1211352f534a54dbc7f8/1533743645902/annurev-resource-100815-095343.pdf#page=22).

## Implications for the Singapore project

1. Keep the current-quarter-temperature model as an explicitly labelled benchmark. The closest quarterly paper supports temperature levels **conditional on a weather history**, not the assertion that one current-quarter variable is always sufficient.
2. For the longer-history quarterly YoY analysis, a useful next specification is unrestricted current temperature plus lags 1–4, with deliberately chosen control histories. Report endpoint and intermediate restrictions jointly. Do not mechanically copy the paper's full date FE: a single national Singapore temperature series would be absorbed.
3. Keep YoY temperature differences and four-quarter exposure as separately interpreted comparisons. A parsimonious joint endpoint-difference/four-quarter-exposure model can compare the two simple channels when the design is identified. Short PM2.5 windows need explicit rank and time-degree-of-freedom checks; extra weather and outcome lags cannot be added freely.
4. Preserve actual-calendar inference for overlapping YoY outcomes. Sector rows do not add distinct weather dates. Choose exposure definitions from the economic question and declared specification plan, rather than statistical significance.
5. Keep the annual analysis alongside quarterly YoY. Annual GDP/VA growth compares annual flow aggregates; quarterly YoY compares two individual quarter flows. They are different outcomes even though each is called annual growth informally.

The saved project already contains current-weather, rolling-year-mean, all-weather YoY-difference and current-plus-year-earlier-weather comparisons. This review does not mean those existing comparisons already implement the unrestricted five-temperature-lag model or every paper's controls. No regressions were re-estimated for this note.

Three primary PDFs inspected locally are cached with source URLs and hashes in `data/raw/literature/temperature_horizon/manifest.json`. Other primary sources and versions are linked above. Subscription access limitations and unconfirmed growth conventions are stated rather than inferred.
