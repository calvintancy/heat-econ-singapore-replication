"""Map every fitted series to explicit unit-root evidence without certifying stationarity."""
from __future__ import annotations
import hashlib
import json
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller,kpss
from build_panel import parse_va
from iip_analysis import source_frame
from verify_sector_cluster import exact_sample
from stationarity_diagnostics import seasonal_adjust

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stationarity'
ALIASES={'g_va_sa_qoq':'y_short','g_va_sa_yoy':'y_yoy','g_va_nsa_yoy':'y_yoy_nsa','g_va_nsa_qoq':'y_short_nsa','g_iip_sa':'y_short','g_iip_nsa':'y_short_nsa'}
OUTCOMES=['y_short','y_yoy','y_yoy_nsa','y_log_level','y_raw_level','y_yoy_pct','y_yoy_pct_nsa','y_short_nsa']
BASE=['temp_c','rain_100mm','rh_pct']
SUPPORT={'stationarity-compatible','trend-stationarity-compatible'}
LABELS={'stationarity-compatible':'Stationarity-compatible','trend-stationarity-compatible':'Trend-stationarity-compatible',
 'unit-root-concern':'Unit-root concern','mixed':'Mixed evidence','inconclusive':'Inconclusive','sensitivity-dependent':'Lag-sensitive evidence',
 'unavailable-gaps':'Not established · calendar gaps','unavailable-short':'Not established · short history','unavailable':'Tests unavailable'}


def canonical(context):
    freq='M' if context=='iip-monthly' else 'Q';period='month' if freq=='M' else 'quarter';h=12 if freq=='M' else 4
    if context=='va':
        f=parse_va('va_sa').merge(parse_va('va_nsa'),on=['sector_id','sector','quarter'],validate='one_to_one').rename(columns={'va_sa':'output_sa','va_nsa':'output_nsa'})
    else:
        f=source_frame('iip_sa','M').merge(source_frame('iip_nsa','M'),on=['sector_id','sector','month'],validate='one_to_one').rename(columns={'iip_sa':'output_sa','iip_nsa':'output_nsa'})
        if freq=='Q':
            f['quarter']=f.month.dt.asfreq('Q')
            f=f.groupby(['sector_id','sector','quarter'],as_index=False).agg(output_sa=('output_sa','mean'),output_nsa=('output_nsa','mean'),months=('month','size'))
            f=f[f.months.eq(3)].copy()
    f=f.sort_values(['sector_id',period]);log=np.log(f.output_sa)
    f['y_log_level']=100*log;f['y_raw_level']=f.output_sa
    f['y_short']=100*log.groupby(f.sector_id).diff();f['y_yoy']=100*log.groupby(f.sector_id).diff(h)
    f['y_short_nsa']=100*np.log(f.output_nsa).groupby(f.sector_id).diff()
    f['y_yoy_pct']=100*(f.output_sa/f.output_sa.groupby(f.sector_id).shift(h)-1)
    f['y_yoy_nsa']=100*np.log(f.output_nsa).groupby(f.sector_id).diff(h)
    f['y_yoy_pct_nsa']=100*(f.output_nsa/f.output_nsa.groupby(f.sector_id).shift(h)-1)
    w=pd.read_csv(ROOT/f'data/processed/weather_{"monthly" if freq=="M" else "quarterly"}.csv')
    w=w.rename(columns={'temperature':'temp_c','rainfall':'rain_mm','humidity':'rh_pct'});w[period]=pd.PeriodIndex(w[period],freq=freq)
    w['rain_100mm']=w.rain_mm/100;w=w.sort_values(period).set_index(period)
    for c in BASE:
        label='temp' if c=='temp_c' else c
        w[label+'_year_lag']=w[c].shift(h);w[label+'_year_change']=w[c]-w[c].shift(h)
        w[label+'_year_mean']=w[c].rolling(h,min_periods=h).mean()
    for lag in range(1,h+1): w['temperature_lag_'+str(lag)]=w.temp_c.shift(lag)
    c=pd.read_csv(ROOT/f'data/processed/extended_controls/{"monthly" if freq=="M" else "quarterly"}_controls.csv')
    c[period]=pd.PeriodIndex(c[period],freq=freq)
    w=w.reset_index().merge(c[[period,'wind_mps','pm25_10ug','wind_coverage','pm_coverage']],on=period,how='left',validate='one_to_one')
    f=f.merge(w,on=period,how='inner',validate='many_to_one')
    return f,period,freq,h


def context_for(path):
    text=str(path.relative_to(ROOT/'outputs'))
    if 'iip-monthly' in text or text.startswith('iip/monthly'):return 'iip-monthly'
    if 'iip-quarterly' in text or text.startswith('iip/quarterly'):return 'iip-quarterly'
    return 'va'


def series_name(v):
    if re.fullmatch(r'temp_l\d+',v):return 'temperature_lag_'+v[6:]
    return ALIASES.get(v,v)


def state(adf,kpss_result,det):
    if adf.get('status')!='ok' or kpss_result.get('status')!='ok':return 'unavailable'
    a=adf['p_value']<.05;k=kpss_result['p_value']<.05
    if a and not k:return 'trend-stationarity-compatible' if det=='ct' else 'stationarity-compatible'
    if not a and k:return 'unit-root-concern'
    return 'mixed' if a else 'inconclusive'


def test_pair(values,det,h,legacy=False):
    rows=[]
    for name in ['ADF','KPSS']:
        row={'test':name,'status':'ok','p_value':None,'p_relation':'=','statistic':None,'lags':None,
             'lag_limit':h if legacy else 2*h,'lag_binding':False,'warnings':''}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if name=='ADF':
                    cap=min(h if legacy else 2*h,len(values)//2-(2 if det=='ct' else 1)-1)
                    statistic,p,lags,nobs,critical,aic=adfuller(values,maxlag=cap,regression=det,autolag='AIC')
                    row.update(lag_limit=cap,lag_binding=lags==cap,nobs=int(nobs),aic=float(aic))
                else:
                    statistic,p,lags,critical=kpss(values,regression=det,nlags='auto')
                    row['auto_lags']=int(lags)
                    if legacy and lags>h:statistic,p,lags,critical=kpss(values,regression=det,nlags=h)
                    row['p_relation']='<' if statistic>critical['1%'] else '>' if statistic<critical['10%'] else '='
                row.update(statistic=float(statistic),p_value=float(p),lags=int(lags),critical_5pct=float(critical['5%']),warnings=' | '.join(str(x.message) for x in caught))
        except (ValueError,OverflowError,ZeroDivisionError) as e:row.update(status='error',error=str(e))
        rows.append(row)
    return rows


class Audit:
    def __init__(self):
        self.sources={c:canonical(c) for c in ['va','iip-monthly','iip-quarterly']}
        self.tests=[];self.rows=[];self.series=[];self.coverage=[];self.models=[];self.cache={};self.history={};self.hashes={}

    def diagnostic(self,context,sector,variable,dates,scope,*,quality=True):
        frame,period,freq,h=self.sources[context]
        weather=variable not in OUTCOMES
        unit='common' if weather else str(sector)
        dates=pd.PeriodIndex(dates,freq=freq).sort_values().unique()
        key=(context,unit,variable,tuple(dates.asi8),scope)
        if key in self.cache:return self.cache[key]
        ident='s'+hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        source=frame.drop_duplicates(period) if weather else frame[frame.sector_id.eq(str(sector))]
        source=source.set_index(period).sort_index()
        values=source[variable].reindex(dates).astype(float)
        if variable in ['wind_mps','pm25_10ug'] and quality:
            coverage='wind_coverage' if variable=='wind_mps' else 'pm_coverage'
            values=values.where(source[coverage].reindex(dates).ge(.90))
        name='Common national weather' if weather else source.sector.iloc[0]
        det='ct' if variable in ['y_log_level','y_raw_level'] or (weather and 'rain_' not in variable and 'year_change' not in variable) else 'c'
        gaps=int(dates.max().ordinal-dates.min().ordinal+1-len(dates)) if len(dates) else 0
        missing=int(values.isna().sum())
        common={'diagnostic_id':ident,'context':context,'sector_id':unit,'sector':name,'variable':variable,'kind':'weather' if weather else 'output',
            'scope':scope,'start':str(dates.min()),'end':str(dates.max()),'n_periods':len(dates),'calendar_gaps':gaps,'missing_values':missing,
            'deterministic':det,'seasonal_adjustment':'Estimated season offsets; trend retained' if weather or variable.endswith('_nsa') else 'Official SA output',
            'frequency':freq,'horizon':h,'primary_state':None,'legacy_state':None,'state':None,'test_status':'not_run'}
        if gaps or missing:common['state']='unavailable-gaps'
        elif len(values)<5*h or np.ptp(values.to_numpy())==0:common['state']='unavailable-short'
        else:
            adjusted,offset=seasonal_adjust(values) if weather or variable.endswith('_nsa') else (values,values*0)
            raw=adjusted.to_numpy()
            for spec,legacy in [('primary',False),('legacy_lag_sensitivity',True)]:
                pair=test_pair(raw,det,h,legacy)
                for row in pair:self.tests.append({**common,'specification':spec,**row})
                common['primary_state' if not legacy else 'legacy_state']=state(*pair,det)
                common[('primary_' if not legacy else 'legacy_')+'adf_p']=pair[0]['p_value']
                common[('primary_' if not legacy else 'legacy_')+'kpss_p']=pair[1]['p_value']
                common[('primary_' if not legacy else 'legacy_')+'kpss_relation']=pair[1]['p_relation']
                common[('primary_' if not legacy else 'legacy_')+'adf_lags']=pair[0]['lags']
                common[('primary_' if not legacy else 'legacy_')+'kpss_lags']=pair[1]['lags']
                common[('primary_' if not legacy else 'legacy_')+'adf_binding']=pair[0]['lag_binding']
            primary,legacy=common['primary_state'],common['legacy_state']
            common['state']=primary if primary==legacy else 'sensitivity-dependent'
            common['test_status']='completed' if primary!='unavailable' and legacy!='unavailable' else 'error'
            for date,value,testvalue,season in zip(dates,values,adjusted,offset):
                self.series.append({'diagnostic_id':ident,'period':str(date),'raw_value':float(value),'test_value':float(testvalue),'seasonal_component':float(season)})
        common['label']=LABELS[common['state']];self.rows.append(common);self.cache[key]=common
        return common

    def background(self,context,sector,variable):
        key=(context,'common' if variable not in OUTCOMES else sector,variable)
        if key in self.history:return self.history[key]
        frame,period,freq,h=self.sources[context]
        group=frame.drop_duplicates(period) if variable not in OUTCOMES else frame[frame.sector_id.eq(sector)]
        group=group.dropna(subset=[variable]);dates=group[period].sort_values().unique()
        # These are full source histories, not evidence for a screened or COVID-excluded sample.
        result=self.diagnostic(context,sector,variable,dates,'available_history')
        self.history[key]=result
        pre=dates[dates.year<2020]
        if len(pre):self.diagnostic(context,sector,variable,pre,'pre_2020_history')
        return result

    def register(self,path,model,outcome,sample,xvars,tag='pooled'):
        context=context_for(path);frame,period,freq,h=self.sources[context]
        if 'sector_id' not in sample:raise ValueError(path)
        sample=sample.copy();sample.sector_id=sample.sector_id.astype(str);sample[period]=pd.PeriodIndex(sample[period],freq=freq)
        relative=str(path.relative_to(ROOT));variables=[series_name(outcome)]+[series_name(x) for x in xvars]
        assert all(v in frame for v in variables),(relative,model,variables)
        unit_states=[];nchecks=0
        for unit,g in sample.groupby('sector_id',sort=True):
            dates=g[period].sort_values().unique();result=[]
            for variable in variables:
                row=self.diagnostic(context,unit,variable,dates,'exact_estimation_window',quality=False)
                history=self.background(context,unit,variable)
                self.coverage.append({'source_csv':relative,'model':model,'outcome':series_name(outcome),'sector_id':unit,'sector':g.sector.iloc[0],
                    'variable':variable,'role':'Y' if variable==variables[0] else 'X','diagnostic_id':row['diagnostic_id'],'exact_state':row['state'],
                    'background_id':history['diagnostic_id'],'background_state':history['state'],'family':tag})
                result.append(row);nchecks+=1
            y=result[0];xs=result[1:]
            allsupport=all(r['state'] in SUPPORT for r in result)
            role='Screen-compatible · growth' if allsupport and variables[0] in ['y_short','y_yoy','y_yoy_nsa','y_yoy_pct','y_yoy_pct_nsa','y_short_nsa'] else 'Exploratory · stationarity not established'
            if variables[0] in ['y_log_level','y_raw_level']:role='Descriptive levels · not primary growth evidence'
            unit_states.append({'sector_id':unit,'y_state':y['state'],'all_variables_compatible':allsupport,'x_issues':sum(r['state'] not in SUPPORT for r in xs),'analysis_role':role})
        self.models.append({'source_csv':relative,'model':model,'outcome':series_name(outcome),'family':tag,'context':context,'n_units':len(unit_states),
            'compatible_y_units':sum(r['y_state'] in SUPPORT for r in unit_states),'compatible_all_variables_units':sum(r['all_variables_compatible'] for r in unit_states),
            'variable_checks':nchecks,'any_calendar_gaps':any(g[period].nunique()!=g[period].max().ordinal-g[period].min().ordinal+1 for _,g in sample.groupby('sector_id')),
            'assessment':'Screen-compatible on this window' if all(r['all_variables_compatible'] for r in unit_states) else 'Stationarity not established for every series',
            'analysis_role':'Descriptive levels' if variables[0] in ['y_log_level','y_raw_level'] else 'Growth · stationarity screen compatible' if all(r['all_variables_compatible'] for r in unit_states) else 'Exploratory growth · stationarity caution'})

    def inventory(self):
        paths=sorted((ROOT/'outputs').rglob('regression_results.csv'))+sorted((ROOT/'outputs').rglob('distributed_lag_results.csv'))
        for path in paths:
            if 'visitor_arrivals' in path.parts:continue  # separate visitor-frequency audit
            if 'formula' not in pd.read_csv(path,nrows=1):continue
            data=pd.read_csv(path,dtype={'sector_id':str})
            self.hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            context=context_for(path);columns=list(self.sources[context][0].columns)+list(ALIASES)+[f'temp_l{i}' for i in range(1,13)]
            for model,rows in data.groupby('model',sort=False):
                formula=rows.formula.iloc[0];outcome=formula.split('~')[0].strip()
                xvars=[x for x in columns if x not in OUTCOMES and x not in ['output_sa','output_nsa'] and x!=outcome and re.search(r'(?<![\w])'+re.escape(x)+r'(?![\w])',formula.split('~')[1])]
                xvars=list(dict.fromkeys(xvars))
                # Deterministic effects/trends are not stochastic weather regressors.
                xvars=[x for x in xvars if x in BASE or x in ['temp_c_unweighted','wind_mps','pm25_10ug'] or 'year_' in x or x.startswith('temperature_lag_') or re.fullmatch(r'temp_l\d+',x)]
                sample=exact_sample(path,model)
                self.register(path,model,outcome,sample,xvars)
        for frequency,context,period in [('monthly','iip-monthly','month'),('quarterly','iip-quarterly','quarter')]:
            path=ROOT/f'outputs/iip/{frequency}/aggregate_results.csv'
            rows=pd.read_csv(path);sample=pd.read_csv(ROOT/f'data/processed/iip/aggregate_{frequency}.csv',dtype={'sector_id':str})
            base=pd.read_csv(ROOT/f'outputs/iip/{frequency}/baseline_sample.csv')
            sample=sample[sample[period].between(base[period].min(),base[period].max())]
            self.register(path,'A1','g_iip_sa',sample,BASE,'national_IIP_original')
        # Heterogeneity and separate fits use these exact common samples. All H1/H2/H3/R1 absolute source series are audited.
        for context in ['va','iip-monthly','iip-quarterly']:
            for controls in ['weather','full']:
                folder=ROOT/'outputs/heterogeneity'/context/controls
                sample=pd.read_csv(folder/'common_sample.csv',dtype={'sector_id':str});terms=BASE+(['wind_mps','pm25_10ug'] if controls=='full' else [])
                slopes=pd.read_csv(folder/'sector_slopes.csv')
                for (model,outcome),_ in slopes.groupby(['model','outcome'],sort=False):
                    selected=sample[~sample.year.between(2020,2022)] if model=='H3X' else sample
                    self.register(folder/'sector_slopes.csv',model,outcome,selected,terms,'sector_interactions_and_equivalent_separate_series')
        # Main detailed-sector FE interactions share the exact Y/X calendar with their separate-series robustness.
        for path in sorted((ROOT/'outputs/granular_interactions').glob('*/*/sector_slopes.csv')):
            data=pd.read_csv(path,dtype={'sector_id':str})
            sample=pd.read_csv(path.parent/'common_sample.csv',dtype={'sector_id':str})
            terms=BASE+(['wind_mps','pm25_10ug'] if path.parent.name!='long' else [])
            self.hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for (model,outcome),rows in data.groupby(['model','outcome'],sort=False):
                self.register(path,model,outcome,sample,terms,'detailed_sector_FE_interactions_main')
        # Detailed separate-series fits share the same outcome/sample as their corresponding pooled model; record explicit coverage links.
        for path in sorted((ROOT/'outputs/granular').rglob('separate_sector_coefficients.csv')):
            data=pd.read_csv(path,dtype={'sector_id':str})
            for model,rows in data.groupby('model',sort=False):
                sample=pd.read_csv(path.parent/'samples'/f'{model}.csv',dtype={'sector_id':str})
                terms=BASE+(['wind_mps','pm25_10ug'] if path.parent.name!='long' else [])
                self.register(path,model,rows.outcome.iloc[0],sample,terms,'individual_sector_time_series')


def run_stationarity_audit():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps({'status':'running','scope':'Expanded stationarity audit'})+'\n')
    audit=Audit();audit.inventory()
    for path in sorted((ROOT/'data/raw').glob('*')):
        if path.is_file() and path.suffix in ['.json','.csv']:
            audit.hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT/'data/processed/extended_controls').glob('*_controls.csv')):
        audit.hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    frames={'diagnostics':pd.DataFrame(audit.rows),'tests':pd.DataFrame(audit.tests),'series':pd.DataFrame(audit.series),
        'model_variable_coverage':pd.DataFrame(audit.coverage),'model_assessments':pd.DataFrame(audit.models)}
    for name,frame in frames.items():frame.to_csv(OUT/f'{name}.csv',index=False)
    counts=frames['diagnostics'].groupby(['scope','kind','variable','state']).size().rename('n_series').reset_index()
    counts.to_csv(OUT/'evidence_counts.csv',index=False)
    summary={'status':'computed_not_a_stationarity_certificate','diagnostics':len(audit.rows),'test_rows':len(audit.tests),
        'model_outcomes':len(audit.models),'model_variable_links':len(audit.coverage),'primary':'ADF AIC with 2h cap; uncapped automatic KPSS',
        'sensitivity':'Legacy ADF h cap and KPSS automatic bandwidth capped at h','minimum_history':'Five seasonal cycles; no compressed gaps',
        'supported_states':list(SUPPORT),'labels':LABELS,'coefficient_inputs_sha256':audit.hashes,
        'scope_note':'Marginal ordinary zero-frequency unit-root screens; not seasonal-unit-root, cointegration, break, residual-stationarity or causal-validity certification.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Stationarity audit computed: {len(audit.rows)} distinct series/windows, {len(audit.tests)} tests, {len(audit.models)} model/outcome groups.',flush=True)
    return summary


if __name__=='__main__':run_stationarity_audit()
