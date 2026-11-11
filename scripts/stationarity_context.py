"""Expose evidence flags beside saved regressions without changing estimates."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from stationarity_audit import LABELS,SUPPORT
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stationarity'


@lru_cache(None)
def _data():
    return pd.read_csv(OUT/'model_assessments.csv',dtype={'model':str}),pd.read_csv(OUT/'model_variable_coverage.csv',dtype={'sector_id':str})


def clear_cache():_data.cache_clear()


def model_assessment(path,model,outcome=None):
    relative=str(Path(path).resolve().relative_to(ROOT));models,coverage=_data()
    rows=models[models.source_csv.eq(relative)&models.model.eq(model)]
    if outcome is not None:rows=rows[rows.outcome.eq(outcome)]
    if len(rows)!=1:raise ValueError(('Missing/ambiguous stationarity mapping',relative,model,outcome,len(rows)))
    row=rows.iloc[0].to_dict()
    row['y_label']=f"{int(row['compatible_y_units'])}/{int(row['n_units'])} compatible"
    row['all_label']=f"{int(row['compatible_all_variables_units'])}/{int(row['n_units'])} Y + X compatible"
    links=coverage[coverage.source_csv.eq(relative)&coverage.model.eq(model)&coverage.outcome.eq(row['outcome'])]
    ys=links[links.role.eq('Y')]
    if ys.exact_state.str.startswith('unavailable').all():
        reason='calendar gaps' if ys.exact_state.eq('unavailable-gaps').all() else 'short/gapped history'
        row['y_label']='Not tested · '+reason
        row['all_label']='Not established · tests unavailable'
    return row


def unit_assessment(path,model,sector_id,outcome=None):
    relative=str(Path(path).resolve().relative_to(ROOT));_,coverage=_data()
    rows=coverage[coverage.source_csv.eq(relative)&coverage.model.eq(model)&coverage.sector_id.eq(str(sector_id))]
    if outcome is not None:rows=rows[rows.outcome.eq(outcome)]
    y=rows[rows.role.eq('Y')]
    if len(y)!=1:raise ValueError(('Missing/ambiguous unit stationarity mapping',relative,model,sector_id,outcome,len(y)))
    xs=rows[rows.role.eq('X')];state=y.iloc[0].exact_state;issues=int((~xs.exact_state.isin(SUPPORT)).sum())
    x_label='Not tested · calendar gaps' if len(xs) and xs.exact_state.eq('unavailable-gaps').all() else ('Not tested · short/gapped history' if len(xs) and xs.exact_state.str.startswith('unavailable').all() else f'{issues}/{len(xs)} controls/exposures unresolved' if issues else 'All X screens compatible')
    return {'y_state':state,'y_label':LABELS[state],'x_issues':issues,'n_x':len(xs),
        'x_label':x_label,
        'background_label':LABELS[y.iloc[0].background_state],
        'all_compatible':state in SUPPORT and not issues}
