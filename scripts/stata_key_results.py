"""Prepare a fixed, documented 99-target Stata/reghdfe heat check.

No existing analytical input or result is written. Execute the generated master
do-file through stata_mcp_client.py, then independently run the Python port.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stata_key_results'
CONTEXTS=['aggregate_va','va12','va20','iip-monthly-total','iip-monthly','iip-monthly-detailed',
          'iip-quarterly-total','iip-quarterly','iip-quarterly-detailed']
DETAILED=['va20','iip-monthly-detailed','iip-quarterly-detailed']
NATIONAL=['aggregate_va','iip-monthly-total','iip-quarterly-total']
VARIANTS=['all_eligible','event_dates_excluded','strict_growth_window_clean']
BASE=['temp_c','rain_100mm','rh_pct'];ADDED=['wind_mps','pm25_10ug']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def target_matrix():
    requests=[]
    def add(context,outcome,controls,variant,structure='common'):
        row=dict(context=context,outcome=outcome,control_set=controls,sample_variant=variant,structure=structure)
        row['target_id']='sk_'+hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
        assert not any(old['target_id']==row['target_id'] for old in requests)
        requests.append(row)
    for context in CONTEXTS:
        for outcome in ['y_short','y_yoy_nsa']:
            for variant in VARIANTS:add(context,outcome,'long',variant)
    for context in CONTEXTS:add(context,'y_yoy_nsa','full90',VARIANTS[0])
    for context in DETAILED:
        for variant in VARIANTS[1:]:add(context,'y_yoy_nsa','full90',variant)
        for controls in ['long','full90']:add(context,'y_yoy_nsa',controls,VARIANTS[0],'H1')
    for context in NATIONAL:add(context,'y_yoy','long',VARIANTS[0])
    for context in CONTEXTS:add(context,'y_short','full90',VARIANTS[0])
    for context in ['va12','va20','iip-monthly-detailed']:
        for variant in VARIANTS[1:]:add(context,'y_short','full90',variant)
    for context in DETAILED:
        for controls in ['long','full90']:add(context,'y_short',controls,VARIANTS[0],'H1')
    assert len(requests)==99
    return requests


MATA=r'''
version 17
mata:
void sk_write(string scalar path, real matrix A) {
    real scalar fh,i
    if(fileexists(path)) unlink(path)
    fh=fopen(path,"w")
    assert(fh>=0)
    for(i=1;i<=rows(A);i++) fput(fh,invtokens(strofreal(A[i,.],"%24.17g"),","))
    fclose(fh)
}
void sk_profiles(string scalar path, real matrix V, real colvector b, real matrix L, real colvector codes) {
    real matrix D
    D=L;D[.,1]=D[.,1]:-1
    sk_write(path,(codes,L*b,diagonal(L*V*L'),J(rows(L),1,b[1]),D*b,
        J(rows(L),1,V[1,1]),diagonal(D*V*D'),D*V[.,1]))
}
void sk_post(string scalar out, string scalar names, real scalar h) {
    real matrix W,A,U,Vt,G,Z,SC,CAL,grid,M,cross,L,Vnative
    real colvector u,t,q,yr,r,s,uv,tv,qv,yv,scale,b,codes
    real scalar n,k,p,i,j,lag,span,fac,rankX,rr
    string rowvector cols
    cols=tokens(names)
    W=st_data(.,cols,"__sample")
    u=st_data(.,"industry_id","__sample")
    t=st_data(.,"time","__sample")
    q=st_data(.,"season","__sample")
    yr=st_data(.,"year","__sample")
    r=st_data(.,"__resid","__sample")
    n=rows(W);p=cols(W)
    uv=uniqrows(sort(u,1));tv=uniqrows(sort(t,1));qv=uniqrows(sort(q,1));yv=uniqrows(sort(yr,1))
    b=st_matrix("e(b)")[1,1..p]'
    assert(all(st_matrixcolstripe("e(b)")[1..p,2]:==cols'))
    L=J(1,p,0);L[1,1]=1;codes=0
    if(any(strpos(cols,"d"):==1)) {
        L=J(rows(uv),p,0);L[.,1]=J(rows(uv),1,1);codes=uv
        for(i=2;i<=rows(uv);i++) L[i,selectindex(cols:=="d"+strofreal(i-1,"%03.0f"))]=1
    }
    A=J(n,0,0)
    for(i=1;i<=rows(uv);i++) for(j=1;j<=rows(qv);j++) A=(A,(u:==uv[i]):*(q:==qv[j]))
    for(i=2;i<=rows(yv);i++) A=(A,yr:==yv[i])
    A=(A,W);k=cols(A)
    scale=sqrt(colsum(A:^2))'
    svd(A:/scale',U,s,Vt)
    rankX=sum(s:>s[1]*max((n,k))*epsilon(1))
    assert(rankX==k)
    /* Design-only inverse; ALL fitted coefficients and residuals came from
       reghdfe. No Mata regression on y is used in this production chain. */
    G=(Vt'*diag(1:/s)*U'):/scale
    G=G[(k-p+1)..k,.]
    Z=G':*r
    sk_write(out+"/hc0.csv",quadcross(Z,Z))
    Vnative=st_matrix("e(V)")[1..p,1..p]
    sk_write(out+"/native_beta.csv",b)
    sk_write(out+"/native_vcov.csv",Vnative)
    sk_profiles(out+"/profiles_native.csv",Vnative,b,L,codes)
    sk_write(out+"/geometry.csv",(n,k,p,rows(tv),rows(uv),rankX,max(abs(G*W-I(p)))))
    if(rows(uv)>1) {
        SC=J(rows(uv),p,0)
        for(i=1;i<=rows(uv);i++) SC[i,.]=colsum(select(Z,u:==uv[i]))
        M=quadcross(SC,SC)
        sk_write(out+"/cluster_unscaled.csv",M)
        fac=rows(uv)/(rows(uv)-1)*(n-1)/(n-k)
        sk_write(out+"/project_cluster.csv",M*fac)
        sk_profiles(out+"/profiles_project_cluster.csv",M*fac,b,L,codes)
    }
    span=max(t)-min(t)+1;grid=J(span,p,0)
    for(i=1;i<=n;i++) grid[t[i]-min(t)+1,.]=grid[t[i]-min(t)+1,.]+Z[i,.]
    fac=rows(tv)/(rows(tv)-1)*(n-1)/(n-k)
    for(j=1;j<=2;j++) {
        M=quadcross(grid,grid)
        for(lag=1;lag<=j*h;lag++) if(lag<span) {
            cross=quadcross(grid[(lag+1)..span,.],grid[1..(span-lag),.])
            M=M+(1-lag/(j*h+1))*(cross+cross')
        }
        sk_write(out+"/calendar_h"+strofreal(j)+".csv",M*fac)
        sk_profiles(out+"/profiles_calendar_h"+strofreal(j)+".csv",M*fac,b,L,codes)
    }
}
end
'''


def prepare_stata_key_results():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['data','do','native','native_tables','mcp']:(OUT/name).mkdir(exist_ok=True)
    policy_path=ROOT/'notes/crisis_sensitivity_policy.json';policy=json.loads(policy_path.read_text())
    hashes={rel(policy_path):sha(policy_path)}
    for path in [ROOT/'outputs/pyfixest/fit_log.csv',ROOT/'outputs/pyfixest/estimator_manifest.json']:
        hashes[rel(path)]=sha(path)
    controls={}
    for frequency,period,freq in [('monthly','month','M'),('quarterly','quarter','Q')]:
        path=ROOT/f'data/processed/extended_controls/{frequency}_controls.csv';frame=pd.read_csv(path,float_precision='round_trip')
        controls[frequency]=frame.set_index(period);hashes[rel(path)]=sha(path)
    requests=target_matrix();data_paths={}
    for number,row in enumerate(requests,1):
        context=row['context'];frequency='monthly' if 'monthly' in context else 'quarterly'
        freq,period,h=('M','month',12) if frequency=='monthly' else ('Q','quarter',4)
        parent_path=ROOT/f'outputs/crisis_sensitivity/parents/{context}.csv'
        hashes[rel(parent_path)]=sha(parent_path)
        parent=pd.read_csv(parent_path,dtype={'sector_id':str},float_precision='round_trip')
        parent=parent.sort_values(['time','sector_id']).reset_index(drop=True)
        frame=parent.copy();original_periods=frame.time.nunique()
        if row['control_set']=='full90':
            c=controls[frequency]
            for field in ADDED+['eligible_90','wind_coverage','pm_coverage']:frame[field]=c[field].reindex(frame[period]).to_numpy()
            frame=frame.loc[frame.eligible_90.eq(True)].copy()
        else:
            frame['wind_mps']=np.nan;frame['pm25_10ug']=np.nan
        current_dates=pd.PeriodIndex(frame[period],freq=freq)
        bad=np.zeros(len(frame),bool)
        for event in policy['events']:
            start=pd.Period(event['start'],freq='M').asfreq(freq);end=pd.Period(event['end'],freq='M').asfreq(freq)
            if row['sample_variant']=='event_dates_excluded':bad |= (current_dates>=start)&(current_dates<=end)
            elif row['sample_variant']=='strict_growth_window_clean':bad |= (current_dates-h<=end)&(current_dates>=start)
        frame=frame.loc[~bad].copy().reset_index(drop=True)
        frame['period']=frame[period];frame['row_id']=np.arange(1,len(frame)+1)
        columns=['sector_id','sector','period','time','year','season','row_id','y_short','y_yoy_nsa','y_yoy']+BASE+ADDED
        frame=frame[columns]
        path=OUT/'data'/f"{context}__{row['control_set']}__{row['sample_variant']}.csv"
        key=str(path)
        if key not in data_paths:frame.to_csv(path,index=False);data_paths[key]=sha(path)
        else:assert data_paths[key]==sha(path)
        sectors=sorted(frame.sector_id.unique());weather=BASE+(ADDED if row['control_set']=='full90' else [])
        aliases={f'x{i:03}':term for i,term in enumerate(weather)}
        if row['structure']=='H1':aliases.update({f'd{i:03}':f'temp_c:sector[{sid}]' for i,sid in enumerate(sectors[1:],1)})
        row.update(order=number,frequency=frequency,horizon=h,source_parent_csv=rel(parent_path),source_parent_sha256=sha(parent_path),
            sample_csv=rel(path),sample_sha256=sha(path),n_obs=len(frame),n_periods=frame.time.nunique(),n_sectors=len(sectors),
            start=str(frame.period.min()),end=str(frame.period.max()),parent_n_periods=original_periods,
            mean_y=float(frame[row['outcome']].mean()),sd_y=float(frame[row['outcome']].std()),
            reference_sector_id=sectors[0],sector_ids=sectors,weather_terms=weather,regressor_aliases=aliases,
            formula=row['outcome']+' ~ temp_c'+(' + temp_c×industry deviations' if row['structure']=='H1' else '')+' + '+' + '.join(weather[1:])+' + industry×season FE + common year FE',
            strict_rule='Every period in [t−h,t] avoids fixed crisis blocks; same mask for short SA, YoY NSA and YoY SA',
            main_inference='Sector CR1 with full LSDV K' if row['structure']=='common' and len(sectors)>1 else 'Actual-calendar Bartlett HAC/DK h with full LSDV K')
    (OUT/'specifications.json').write_text(json.dumps(requests,indent=2,ensure_ascii=False)+'\n')
    pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in requests]).to_csv(OUT/'specifications.csv',index=False)
    (OUT/'policy.json').write_text(json.dumps(policy,indent=2,ensure_ascii=False)+'\n')
    (OUT/'do/postprocess.mata.do').write_text(MATA)
    do=['version 17','clear all','set more off','set type double','set linesize 160',
        f'cd "{ROOT}"',f'log using "{OUT}/native/master_stata_output.txt", text replace','which reghdfe','which ftools','which esttab',
        f'do "{OUT}/do/postprocess.mata.do"',
        f'file open meta using "{OUT}/native/runs.csv", write replace',
        'file write meta "target_id,run_variant,n_obs,n_eligible,n_parameters_reported,df_m,df_a,df_a_nested,df_r,n_clusters,n_singletons,r_squared,converged" _n']
    for row in requests:
        target=row['target_id'];folder=OUT/'native'/target;folder.mkdir(exist_ok=True)
        do += [f'display "TARGET {row["order"]}/99 {target}"',f'import delimited using "{ROOT/row["sample_csv"]}", clear varnames(1) stringcols(1) bindquote(strict)',
            'isid row_id','isid sector_id time','egen long industry_id=group(sector_id)',f'generate double y={row["outcome"]}']
        for i,term in enumerate(row['weather_terms']):do += [f'quietly summarize {term}',f'generate double x{i:03}={term}-r(mean)',f'label variable x{i:03} "{term}"']
        if row['structure']=='H1':
            for i,sid in enumerate(row['sector_ids'][1:],1):do += [f'generate double d{i:03}=x000*(industry_id=={i+1})',f'label variable d{i:03} "temp_c x {sid}"']
        rhs=' '.join(row['regressor_aliases']);vce='cluster industry_id' if row['n_sectors']>1 else 'robust'
        command=f'reghdfe y {rhs}, absorb(industry_id#season year) vce({vce}) tolerance(1e-12) iterate(16000) residuals(__resid)'
        do += [command,f'estimates store t{row["order"]:03}','generate byte __sample=e(sample)',f'local needs_exact=(e(N)<{row["n_obs"]})']
        for run_variant in ['native_default','exact_eligible']:
            dest=folder/run_variant;dest.mkdir(exist_ok=True)
            if run_variant=='exact_eligible':do += ['if `needs_exact\' {','drop __resid __sample',command+' keepsingletons','generate byte __sample=e(sample)',f'assert e(N)=={row["n_obs"]}']
            do += [f'export delimited row_id time industry_id __sample __resid using "{dest}/sample_residuals.csv", replace',
                f'mata: sk_post("{dest}","{rhs}",{row["horizon"]})',
                'local ng=cond("`e(vce)\'"=="cluster",e(N_clust),0)',
                f'file write meta "{target},{run_variant}," %21.17g (e(N)) ",{row["n_obs"]},{len(row["regressor_aliases"])} ," %21.17g (e(df_m)) "," %21.17g (e(df_a)) "," %21.17g (e(df_a_nested)) "," %21.17g (e(df_r)) "," %21.17g (`ng\') "," %21.17g (e(num_singletons)) "," %21.17g (e(r2)) "," %21.17g (e(converged)) _n']
            if run_variant=='exact_eligible':do += ['}']
    do += ['file close meta']
    groups={}
    for row in requests:groups.setdefault((row['context'],row['outcome'],row['control_set'],row['structure']),[]).append(row)
    for key,group in groups.items():
        path=OUT/'native_tables'/('__'.join(key)+'.html');stored=' '.join(f't{row["order"]:03}' for row in group)
        titles=' '.join('"'+row['sample_variant']+'"' for row in group)
        display='cells(b(fmt(6))) nostar' if key[-1]=='H1' else 'b(6) se(6) star(* 0.10 ** 0.05 *** 0.01)'
        note='Native H1 unit-cluster covariance is highly singular; coefficients only. Joint calendar inference is exported separately.' if key[-1]=='H1' else 'Native default sample and covariance; project-calendar corrections are exported separately.'
        do += [f'esttab {stored} using "{path}", replace html {display} stats(N r2 df_a df_a_nested, labels("N" "R-squared" "Absorbed df" "Nested absorbed df")) mtitles({titles}) title("Native reghdfe: {" / ".join(key)}") addnotes("{note}")']
    do += [f'file open done using "{OUT}/native/completed.txt", write replace','file write done "99 target analyses completed" _n','file close done','log close']
    (OUT/'do/key_results.do').write_text('\n'.join(do)+'\n')
    # Record package source versions without reading or copying license files.
    ado=Path.home()/'Library/Application Support/Stata/ado/plus'
    packages={}
    for name in ['reghdfe','ftools','esttab','estout']:
        path=ado/name[0]/f'{name}.ado'
        if path.exists():packages[name]={'path':str(path),'sha256':sha(path),'header':path.read_text(errors='replace').splitlines()[0]}
    manifest={'status':'prepared_not_executed','targets':99,'selection':'Fixed protocol; no outcome, sign or significance selection',
        'inputs_sha256':hashes,'prepared_sample_sha256':{rel(path):value for path,value in data_paths.items()},
        'packages':packages,'native_estimator':'reghdfe; Stata17/IC detected through real MCP stdio',
        'native_vce':'Default reghdfe robust/sector-cluster conventions retained; exact-eligible keepsingletons fit only when default changes rows',
        'project_covariance':'Mata design-only SVD inverse and native reghdfe residuals, full-LSDV-K CR1 and actual-calendar Bartlett h/2h; H1 total main+deviation full covariance',
        'do_sha256':sha(OUT/'do/key_results.do'),'mata_sha256':sha(OUT/'do/postprocess.mata.do'),
        'primary_method_urls':['https://scorreia.com/help/reghdfe.html','https://scorreia.com/software/reghdfe/faq.html','https://www.stata.com/manuals/m-5svd.pdf']}
    assert all(sha(ROOT/path)==digest for path,digest in hashes.items())
    (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(f'Prepared {len(requests)} Stata targets and {len(data_paths)} frozen eligible datasets.',flush=True)
    return manifest


if __name__=='__main__':prepare_stata_key_results()
