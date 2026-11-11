"""Reference-coded temperature interactions and reported absolute sector effects."""
import numpy as np
from scipy.stats import t
from calendar_hac import cov_nw_groupsum
from pyfixest_backend import fit,ESTIMATOR_VERSION


def sector_temperature_slopes(sample,outcome,weather,fe,period='quarter',lags=4,label='Sector FE temperature impacts'):
    formula=f'{outcome} ~ temp_c + C(sector_id):temp_c + {weather} + {fe}'
    model=fit(sample,formula,label)
    time=(sample.time-sample.time.min()).to_numpy(dtype=int)
    covariance=cov_nw_groupsum(model,nlags=lags,time=time,use_correction='cluster')
    sectors=sorted(sample.sector_id.unique());names=sample.drop_duplicates('sector_id').set_index('sector_id').sector.to_dict()
    ref=sectors[0];index=model.model.exog_names;main=index.index('temp_c');df=sample[period].nunique()-1;critical=t.ppf(.975,df)
    rows=[]
    for sector in sectors:
        weights=np.zeros(len(index));weights[main]=1
        delta=0.0
        if sector!=ref:
            name=f'C(sector_id)[T.{sector}]:temp_c';j=index.index(name);weights[j]=1;delta=float(model.params.iloc[j])
        beta=float(weights@model.params.to_numpy());se=float(np.sqrt(weights@covariance@weights))
        rows.append({'model':'H1','label':label,'term':f'C(sector_id)[{sector}]:temp_c','sector_id':sector,'sector':names[sector],
            'coefficient':beta,'std_error':se,'ci_lower':beta-critical*se,'ci_upper':beta+critical*se,
            'p_value':float(2*t.sf(abs(beta/se),df)),'n_obs':len(sample),'n_sectors':len(sectors),f'n_{period}s':sample[period].nunique(),
            'start':str(sample[period].min()),'end':str(sample[period].max()),'r_squared':model.rsquared,'covariance':'Driscoll-Kraay',
            'hac_lags':lags,'inference_df':df,'formula':formula,'estimator':'PyFixest feols','estimator_version':ESTIMATOR_VERSION,
            'temperature_main':float(model.params.iloc[main]),'temperature_interaction':delta,'reference_sector_id':ref,
            'reported_effect':'temperature main + sector interaction; reference interaction normalized to zero'})
    return rows
