"""Detailed VA, manufacturing industries and the official weighted IIP total."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from estimate import FE, estimates, fit
from build_panel import parse_va, BROAD as VA_BROAD
from iip_analysis import source_frame, BROAD as IIP_BROAD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/granular'
OUTCOMES = {'G1': ('y_short', 'One-period log growth (SA)'), 'G2N': ('y_yoy_nsa', 'YoY log growth (NSA)'), 'G2': ('y_yoy', 'YoY log growth (SA)'),
            'L1': ('y_log_level', '100 × log output'), 'L3': ('y_raw_level', 'Output level')}
CONTEXTS = [('va-detailed','VA · 19 detailed sectors','quarterly'),
 ('va-broad-matched','VA · 11 broad sectors, matched dates','quarterly'),
 ('iip-monthly-total','Manufacturing · published weighted total','monthly'),
 ('iip-monthly-detailed','Manufacturing · 18 detailed industries','monthly'),
 ('iip-monthly-broad-matched','Manufacturing · 6 clusters, matched dates','monthly'),
 ('iip-quarterly-total','Manufacturing · published weighted total','quarterly'),
 ('iip-quarterly-detailed','Manufacturing · 18 detailed industries','quarterly'),
 ('iip-quarterly-broad-matched','Manufacturing · 6 clusters, matched dates','quarterly')]


def prepare(frequency: str, va: bool = False) -> pd.DataFrame:
    period, freq, horizon = ('month','M',12) if frequency == 'monthly' else ('quarter','Q',4)
    if va:
        levels = parse_va('va_sa').rename(columns={'va_sa':'output_sa'})
        nsa = parse_va('va_nsa').rename(columns={'va_nsa':'output_nsa'})
        levels = levels.merge(nsa[['sector_id',period,'output_nsa']],on=['sector_id',period],how='left',validate='one_to_one')
    else:
        levels = source_frame('iip_sa','M').rename(columns={'iip_sa':'output_sa'})
        nsa = source_frame('iip_nsa','M').rename(columns={'iip_nsa':'output_nsa'})
        levels = levels.merge(nsa[['sector_id','month','output_nsa']],on=['sector_id','month'],how='left',validate='one_to_one')
        if frequency == 'quarterly':
            levels['quarter'] = levels.month.dt.asfreq('Q')
            levels = levels.groupby(['sector_id','sector','quarter'],as_index=False).agg(output_sa=('output_sa','mean'),output_nsa=('output_nsa','mean'),nsa_count=('output_nsa','count'),months=('month','size'))
            levels = levels[levels.months.eq(3)].copy()
            levels.loc[levels.nsa_count.ne(3),'output_nsa']=np.nan
            levels=levels.drop(columns='nsa_count')
    levels = levels.sort_values(['sector_id',period]).copy()
    log = np.log(levels.output_sa)
    # Differences are computed on full consecutive source histories before any exclusions.
    levels['y_short'] = 100*log.groupby(levels.sector_id).diff()
    levels['y_yoy'] = 100*log.groupby(levels.sector_id).diff(horizon)
    nsa_history=levels.set_index(['sector_id',period]).output_nsa
    nsa_base=nsa_history.reindex(pd.MultiIndex.from_arrays([levels.sector_id,levels[period]-horizon])).to_numpy()
    levels['y_yoy_nsa']=100*(np.log(levels.output_nsa)-np.log(nsa_base))
    assert levels.loc[levels.y_yoy.notna(),'y_yoy_nsa'].notna().all(), 'NSA coverage must preserve existing SA samples'
    levels['y_log_level'] = 100*log
    levels['y_raw_level'] = levels.output_sa
    weather = pd.read_csv(ROOT/f'data/processed/weather_{frequency}.csv')
    weather[period] = pd.PeriodIndex(weather[period],freq=freq)
    weather = weather.rename(columns={'temperature':'temp_c','humidity':'rh_pct','rainfall':'rain_mm'})
    controls = pd.read_csv(ROOT/f'data/processed/extended_controls/{frequency}_controls.csv')
    controls[period] = pd.PeriodIndex(controls[period],freq=freq)
    weather = weather.merge(controls[[period,'wind_mps','pm25_10ug','eligible_90']],on=period,how='left',validate='one_to_one')
    frame = levels.merge(weather,on=period,validate='many_to_one')
    frame['eligible_90'] = frame.eligible_90.astype('boolean').fillna(False).astype(bool)
    frame['rain_100mm'] = frame.rain_mm/100
    frame['year'] = frame[period].dt.year
    frame['season'] = frame[period].dt.month if freq == 'M' else frame[period].dt.quarter
    frame['time'] = frame[period].astype('int64')-pd.Period('1982-01' if freq=='M' else '1982Q1',freq=freq).ordinal
    return frame


def select(frame: pd.DataFrame, key: str, va: bool) -> pd.DataFrame:
    catalog = pd.read_csv(ROOT/('data/processed/sector_catalog.csv' if va else 'data/processed/iip/sector_catalog.csv'),dtype={'sector_id':str})
    detailed = catalog.loc[catalog.detailed_panel,'sector_id'].tolist()
    ids = detailed if key.endswith('detailed') else (VA_BROAD if va else IIP_BROAD)
    if key.endswith('total'):
        ids = ['1']
    subset = frame[frame.sector_id.isin(ids)].dropna(subset=['y_short','y_yoy','y_log_level','y_raw_level']).copy()
    start = subset.groupby('sector_id').time.min().max()
    end = subset.groupby('sector_id').time.max().min()
    if key.endswith('broad-matched'):
        leaves = frame[frame.sector_id.isin(detailed)].dropna(subset=['y_short','y_yoy'])
        start = max(start,leaves.groupby('sector_id').time.min().max())
        end = min(end,leaves.groupby('sector_id').time.max().min())
    subset = subset[subset.time.between(start,end)].sort_values(['sector_id','time']).reset_index(drop=True)
    assert subset.groupby('time').sector_id.nunique().eq(len(ids)).all()
    return subset


def run_granular() -> None:
    prepared = {('va','quarterly'):prepare('quarterly',True),('iip','monthly'):prepare('monthly'),('iip','quarterly'):prepare('quarterly')}
    summary=[]
    for key,label,frequency in CONTEXTS:
        va = key.startswith('va-')
        frame=select(prepared[('va' if va else 'iip',frequency)],key,va)
        period,h=('month',12) if frequency=='monthly' else ('quarter',4)
        for window in ['long','full','exclude']:
            sample=frame.copy() if window=='long' else frame[frame.eligible_90].copy()
            controls=['rain_100mm','rh_pct']+([] if window=='long' else ['wind_mps','pm25_10ug'])
            if window=='exclude': sample=sample[~sample.year.between(2020,2022)].copy()
            sample=sample.sort_values(['sector_id','time']).reset_index(drop=True)
            out=OUT/key/window
            (out/'samples').mkdir(parents=True,exist_ok=True)
            rows=[]; separate=[]
            for model_id,(y,ylabel) in OUTCOMES.items():
                formula=f'{y} ~ temp_c + '+ ' + '.join(controls)+f' + {FE}'
                model=fit(sample,formula,key+' '+window+' '+model_id)
                fitted_rows=estimates(model,sample,model_id,label+' · '+ylabel,formula,['temp_c']+controls,lags=h,period_col=period)
                # A doubled-bandwidth calendar-HAC sensitivity is retained for overlapping YoY outcomes.
                fitted_rows+=estimates(model,sample,model_id,label+' · '+ylabel,formula,['temp_c'],lags=2*h,period_col=period)
                k=model.model.exog.shape[1]
                for row in fitted_rows:
                    row.update(context=key,window=window,outcome=y,mean_y=float(sample[y].mean()),sd_y=float(sample[y].std()),
                        time_residual_df=int(sample[period].nunique()-(pd.get_dummies(sample.drop_duplicates(period)[['season','year']].astype(str),drop_first=True).shape[1]+1+1+len(controls))),
                        output_adjustment='NSA' if y.endswith('_nsa') else 'SA',djo_log_growth_form=model_id in ['G1','G2','G2N'],effect_0p1c=.1*row['coefficient'])
                rows+=fitted_rows
                sample.to_csv(out/'samples'/f'{model_id}.csv',index=False)
                if key.endswith('detailed'):
                    for sector_id,series in sample.groupby('sector_id',sort=True):
                        series=series.reset_index(drop=True)
                        singleformula=f'{y} ~ temp_c + '+' + '.join(controls)+' + C(season) + C(year)'
                        single=fit(series,singleformula,key+' '+sector_id+' '+model_id)
                        for row in estimates(single,series,model_id,ylabel,singleformula,['temp_c'],lags=h,period_col=period):
                            row.update(sector_id=sector_id,sector=series.sector.iloc[0],mean_y=float(series[y].mean()),sd_y=float(series[y].std()),
                                time_residual_df=int(single.df_resid),outcome=y,context=key,window=window)
                            separate.append(row)
                summary.append({'context':key,'label':label,'window':window,'model':model_id,'n_sectors':sample.sector_id.nunique(),
                    'n_periods':sample[period].nunique(),'n_obs':len(sample),'start':str(sample[period].min()),'end':str(sample[period].max()),
                    'mean_y':sample[y].mean(),'sd_y':sample[y].std(),'pooled_parameters':k})
            pd.DataFrame(rows).to_csv(out/'regression_results.csv',index=False)
            if separate: pd.DataFrame(separate).to_csv(out/'separate_sector_coefficients.csv',index=False)
            print(f'Granular {key} {window}: {sample.sector_id.nunique()} sectors × {sample[period].nunique()} periods',flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)


if __name__=='__main__': run_granular()
