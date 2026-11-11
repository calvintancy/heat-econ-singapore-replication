"""Readable sector-level uncertainty and stationarity from the saved inference."""
import html
from significance import significance_label,significance_class,significance_stars,p_format
from stationarity_context import unit_assessment


def sector_readout(frame,baseline_csv,model='S3'):
    rows=[]
    for r in frame.itertuples():
        stationarity=unit_assessment(baseline_csv,model,r.sector_id)
        rows.append(f'<tr data-sector-id="{r.sector_id}" data-p-value="{r.p_value:.17g}"><td>{html.escape(r.sector)}</td><td>{r.coefficient:.3f}{significance_stars(r.p_value)}</td><td>{html.escape(p_format(r.p_value))}</td><td><strong class="sig-badge {significance_class(r.p_value)}">{significance_label(r.p_value)}</strong></td><td>{html.escape(stationarity["y_label"])}</td><td>{html.escape(stationarity["x_label"])}</td></tr>')
    return '<div class="table-scroll"><table><thead><tr><th>Sector</th><th>Temperature β</th><th>HAC p-value</th><th>Pointwise significance</th><th>Y stationarity · exact window</th><th>Weather screens</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p class="table-note">*** p&lt;0.01; ** p&lt;0.05; * p&lt;0.10. Labels use saved unrounded p-values. These are pointwise nominal tests; checking many sectors raises the chance of false positives. Significance does not clear an unresolved stationarity screen. <a href="stationarity.html">All unit-root results →</a></p>'
