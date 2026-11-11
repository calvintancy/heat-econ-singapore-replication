"""Independent sector-score checks of distributed-lag and paired-weather contrasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from verify_sector_cluster import manual_fit, exact_sample, close
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'


def verify_cluster_contrasts():
    records=[]
    sources=[(OUT,'sum_lag_0_to_4'),(OUT/'iip/monthly','sum_lag_0_to_max'),(OUT/'iip/quarterly','sum_lag_0_to_max')]
    for folder,key in sources:
        path=folder/'distributed_lag_results.csv'
        rows=pd.read_csv(path)
        chosen=rows[rows.model.eq('L1') & rows.covariance.eq('Sector cluster')]
        sample=exact_sample(path,'L1')
        records.append(check(sample,chosen.formula.iloc[0],chosen.term.tolist(),json.loads((folder/'distributed_lag_sum.json').read_text())['sector_cluster'],key,str(folder.relative_to(OUT))))
    for context in ['va','iip-monthly','iip-quarterly']:
        folder=OUT/'outcomes'/context/'yoy_weather'
        rows=pd.read_csv(folder/'regression_results.csv')
        sample=pd.read_csv(folder/'common_sample.csv',dtype={'sector_id':str})
        for suffix in ['', 'N']:
            chosen=rows[rows.model.eq('Y4'+suffix) & rows.covariance.eq('Sector cluster')]
            name='temperature_difference_restriction'+('_nsa' if suffix else '')+'.json'
            records.append(check(sample,chosen.formula.iloc[0],['temp_c','temp_year_lag'],json.loads((folder/name).read_text())['sector_cluster'],'sum',context+(' NSA' if suffix else ' SA')))
    result={'status':'passed','contrasts_checked':len(records),'method':'Independent QR and explicit sector score sums; t(S−1) contrast tests','details':records}
    (OUT/'sector_cluster_contrasts_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Verified {len(records)} sector-CR1 linear contrasts independently.')
    return result


def check(sample,formula,terms,saved,key,label):
    independent=manual_fit(sample,formula)
    indices=[independent['terms'].index(term) for term in terms]
    beta=float(independent['beta'][indices].sum())
    se=float(np.sqrt(independent['covariance'][np.ix_(indices,indices)].sum()))
    df=independent['inference_df']; critical=t.ppf(.975,df)
    close(saved[key],beta,label+' contrast beta')
    close(saved['std_error'],se,label+' contrast SE')
    close(saved['p_value'],2*t.sf(abs(beta/se),df),label+' contrast p')
    close([saved['ci_lower'],saved['ci_upper']],[beta-critical*se,beta+critical*se],label+' contrast CI')
    assert saved['inference_df']==df and saved['n_clusters']==independent['n_clusters']
    return {'context':label,'terms':terms,'n_clusters':independent['n_clusters'],'beta_error':float(saved[key]-beta),'se_error':float(saved['std_error']-se)}

if __name__=='__main__': verify_cluster_contrasts()
