"""Main detailed-sector FE model: sector temperature interactions and shared controls."""
from __future__ import annotations
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from heterogeneity import ESTIMATOR_META,OUTCOME_LABELS,project,component_records,covariance,inference,wald
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/granular_interactions'
MODELS={'G1':'y_short','G2':'y_yoy','G2N':'y_yoy_nsa','L1':'y_log_level','L3':'y_raw_level'}
YS=list(MODELS.values())
CONTEXTS=['va-detailed','iip-monthly-detailed','iip-quarterly-detailed']


def projection(sample,terms,label=None):
    result=project(sample,terms,'H1',outcomes=YS,label=label)
    result['nuisance']=result['nuisance_beta']
    return result


def run_granular_interactions():
    summaries=[]
    for context in CONTEXTS:
        monthly='monthly' in context;period='month' if monthly else 'quarter';freq='M' if monthly else 'Q';h=12 if monthly else 4
        for window in ['long','full','exclude']:
            original=ROOT/'outputs/granular'/context/window
            sample=pd.read_csv(original/'samples/G1.csv',dtype={'sector_id':str})
            sample[period]=pd.PeriodIndex(sample[period],freq=freq)
            sample=sample.sort_values(['time','sector_id']).reset_index(drop=True)
            assert sample.groupby(period).sector_id.nunique().eq(sample.sector_id.nunique()).all()
            terms=['temp_c','rain_100mm','rh_pct']+(['wind_mps','pm25_10ug'] if window!='long' else [])
            result=projection(sample,terms,label=f'Detailed FE {context}/{window}/H1')
            folder=OUT/context/window;folder.mkdir(parents=True,exist_ok=True);sample.to_csv(folder/'common_sample.csv',index=False)
            names=sample.drop_duplicates('sector_id').set_index('sector_id').sector.to_dict()
            rows=[];joint=[];pairs=[];covrows=[];allcoef=[];components=[];component_covariances=[]
            common=pd.read_csv(original/'regression_results.csv')
            for j,outcome in enumerate(YS):
                model_id=list(MODELS)[j]
                benchmark=common[common.model.eq(model_id)&common.term.eq('temp_c')].iloc[0]
                mean_error=float(result['beta'][:,j].mean()-benchmark.coefficient)
                assert abs(mean_error)<1e-7,(context,window,outcome,mean_error)
                for name,value in zip(result['nuisance_names'],result['nuisance'][:,j]):
                    allcoef.append({'model':'H1','outcome':outcome,'term':name,'coefficient':float(value),**ESTIMATOR_META})
                beta=result['beta'][:,j]
                for bandwidth in [h,2*h]:
                    v=covariance(result,sample,j,bandwidth)
                    meta={'model':'H1','label':'Main FE · temperature × sector, shared controls and year FE','context':context,'window':window,
                        'outcome':outcome,'outcome_label':OUTCOME_LABELS[outcome],'covariance':'Driscoll-Kraay','hac_lags':bandwidth,'inference_df':result['t']-1,
                        'n_obs':result['n'],'n_sectors':len(beta),'n_periods':result['t'],'n_parameters':result['k'],
                        'r_squared':float(result['r_squared'][j]),'start':str(sample[period].min()),'end':str(sample[period].max()),
                        'formula':outcome+' ~ temp_c * C(sector_id) + '+' + '.join(terms[1:])+' + C(sector_id)*C(season) + C(year)',
                        'estimand':'Absolute sector temperature slope in common-controls FE interaction model',**ESTIMATOR_META}
                    component_rows,component_cov=component_records(result,v,j,meta,names)
                    components+=component_rows;component_covariances+=component_cov
                    for i,sector in enumerate(result['sectors']):
                        series=sample[sample.sector_id.eq(sector)]
                        rows.append({**meta,'sector_id':sector,'sector':names[sector],'mean_y':float(series[outcome].mean()),
                            'sd_y':float(series[outcome].std()),**inference(beta[i],v[i,i],result['t'])})
                        allcoef.append({'model':'H1','outcome':outcome,'term':f'temp_c[{sector}]','coefficient':float(beta[i]),**ESTIMATOR_META}) if bandwidth==h else None
                        for k,other in enumerate(result['sectors']):covrows.append({'model':'H1','outcome':outcome,'hac_lags':bandwidth,'sector_i':sector,'sector_j':other,'covariance':float(v[i,k]),**ESTIMATOR_META})
                    restrictions=np.eye(len(beta))[1:]-np.eye(len(beta))[[0]]
                    # In a balanced common-weather panel, sector differences cancel shared year/control terms.
                    # Their time-series design is temperature + sector-difference seasonal intercepts.
                    # Hence the joint contrast influence rank cannot exceed T−Q−1.
                    bound=result['t']-sample.season.nunique()-1
                    joint.append({**meta,'test':'All detailed-sector temperature slopes equal',**wald(beta,v,restrictions,result['t'],rank_bound=int(bound))})
                    family=[]
                    for i in range(len(beta)):
                        for k in range(i+1,len(beta)):
                            variance=float(v[i,i]+v[k,k]-2*v[i,k]);first=result['sectors'][i];second=result['sectors'][k]
                            family.append({**meta,'first_sector_id':first,'second_sector_id':second,'comparison':names[first]+' − '+names[second],
                                **inference(float(beta[i]-beta[k]),variance,result['t'])})
                    adjusted=multipletests([r['p_value'] for r in family],method='holm')[1]
                    for row,p in zip(family,adjusted):row.update(p_value_holm=float(p),multiplicity_family=f'All {len(family)} sector pairs in this context/window/outcome/bandwidth')
                    pairs+=family
                summaries.append({'context':context,'window':window,'outcome':outcome,'n_sectors':len(beta),'n_periods':result['t'],
                    'n_parameters':result['k'],'mean_sector_slope':float(beta.mean()),'common_slope_benchmark':float(benchmark.coefficient),
                    'mean_slope_benchmark_error':mean_error})
            for name,data in [('sector_slopes',rows),('joint_tests',joint),('pairwise_tests',pairs),('slope_covariances',covrows),('all_coefficients',allcoef),
                              ('interaction_components',components),('component_covariances',component_covariances)]:pd.DataFrame(data).to_csv(folder/(name+'.csv'),index=False)
            print(f'Detailed FE interactions {context}/{window}: {len(result["sectors"])} sectors × {result["t"]} dates; {len(YS)} Y definitions (SA and NSA YoY retained)',flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    (OUT/'methods.json').write_text(json.dumps({'main':'Temperature × sector; sector × season FE; common year FE and shared weather controls',
        'outcomes':{model:OUTCOME_LABELS[outcome] for model,outcome in MODELS.items()},
        'yoy':'G2 retains YoY log growth of seasonally adjusted levels; G2N adds the same annual-horizon log difference of non-seasonally adjusted levels. Temperature remains in current-period levels.',
        'estimator':'Actual PyFixest feols for every outcome, including nuisance coefficients and residuals; explicit full-rank LSDV matrix.',
        'temperature_parameterization':'Temperature main effect plus S−1 nonreference sector interactions. The reference slope is the main effect; every other absolute slope is main plus its interaction. Weather is centered in the explicit design, leaving slope interpretations unchanged with sector-season effects.',
        'component_inference':'interaction_components.csv stores native main/deviation coefficients; component_covariances.csv stores their joint calendar HAC covariance. Total SE uses Var(main)+Var(deviation)+2Cov(main,deviation). Reference has no estimated zero deviation.',
        'robustness':'Separate-series fits allow every control and year effect to vary by sector; their point slopes equal fully interacted H3',
        'inference':'Joint calendar-aware Driscoll–Kraay for sector-specific coefficients, Bartlett h/2h; t(T−1). Common-slope benchmark retains sector CR1.',
        'joint_rank_bound':'For balanced common-weather FE slope contrasts, at most T−Q−1; no singular-covariance omnibus p-value.',
        'pairs':'All S(S−1)/2 pairs, Holm-adjusted within each context/window/outcome/bandwidth; no cherry-picked family.'},indent=2)+'\n')

if __name__=='__main__':run_granular_interactions()
