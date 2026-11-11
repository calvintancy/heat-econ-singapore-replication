"""Select the user-requested unit clustering without discarding HAC benchmarks."""
import pandas as pd


def main_rows(rows: pd.DataFrame, frequency: str) -> pd.DataFrame:
    h = 12 if frequency == 'monthly' else 4
    clustered = rows[rows.covariance.eq('Sector cluster')]
    fallback = rows[rows.covariance.eq('Driscoll-Kraay') & rows.hac_lags.eq(h)]
    # Models without clustered estimates are national/sector-specific time series.
    return pd.concat([clustered, fallback[~fallback.model.isin(clustered.model.unique())]],ignore_index=True)


def djo_form(dependent: str) -> str:
    value = dependent.lower()
    if 'log growth' in value:
        return 'Yes · log difference; frequency adapted'
    if 'growth' in value and ('mom' in value or 'qoq' in value) and 'exact' not in value:
        return 'Yes · log difference; frequency adapted'
    return 'No · level / exact-percent sensitivity'
