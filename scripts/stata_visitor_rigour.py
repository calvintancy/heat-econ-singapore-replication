"""Prepare38 prespecified native reghdfe arrivals checks; no old writes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from stata_key_results import MATA

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stata_key_results/arrivals'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(ROOT))


def arrival_matrix():
    rows=[]
    def add(panel,outcome,controls,variant,model='common',relative_only=False):
        record=dict(panel=panel,context=panel,outcome=outcome,control_set=controls,sample_variant=variant,structure=model,relative_only=relative_only)
        record['target_id']='sa_'+hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
        rows.append(record)
    variants=['all_eligible','event_dates_excluded','strict_growth_window_clean']
    for panel in ['aggregate','region','country']:
        for y in ['y_mom_nsa','y_yoy_nsa']:
            for variant in variants:add(panel,y,'long',variant)
            add(panel,y,'full','all_eligible')
    for y in ['y_mom_nsa','y_yoy_nsa']:
        for variant in variants[1:]:add('country',y,'full',variant)
    for panel in ['region','country']:
        for y in ['y_mom_nsa','y_yoy_nsa']:
            for control in ['long','full']:add(panel,y,control,'all_eligible','H1')
    add('country','y_yoy_nsa','long','all_eligible','H1',True)
    add('region','y_mom_nsa','full','all_eligible','H1',True)
    assert len(rows)==38 and len({row['target_id'] for row in rows})==38
    return rows


def arrivals_mata():
    text=MATA
    text=text.replace('real matrix W,A,U,Vt,G,Z,SC,CAL,grid,M,cross,L,Vnative','real matrix W,A,U,Vt,G,Z,SC,CAL,grid,M,cross,L,Vnative,localA,rank_rows')
    text=text.replace('real colvector u,t,q,yr,r,s,uv,tv,qv,yv,scale,b,codes','real colvector u,t,q,yr,r,s,uv,tv,qv,yv,scale,b,codes,local_scale,local_sv')
    text=text.replace('real scalar n,k,p,i,j,lag,span,fac,rankX,rr','real scalar n,k,p,i,j,lag,span,fac,rankX,rr,kt,ii')
    insertion='''A=(A,W);k=cols(A)
    rank_rows=J(rows(uv),2,.)
    for(ii=1;ii<=rows(uv);ii++) {
        localA=select(A,u:==uv[ii]);local_scale=sqrt(colsum(localA:^2))'
        local_scale=local_scale:+(local_scale:==0)
        local_sv=svdsv(localA:/local_scale')
        rank_rows[ii,.]=(uv[ii],sum(local_sv:>local_sv[1]*max((rows(localA),cols(localA)))*epsilon(1)))
    }
    kt=max(rank_rows[.,2]);assert(rows(tv)>kt)
    sk_write(out+"/time_ranks.csv",rank_rows)'''
    assert 'A=(A,W);k=cols(A)' in text;text=text.replace('A=(A,W);k=cols(A)',insertion)
    text=text.replace('rankX,max(abs(G*W-I(p)))))','rankX,max(abs(G*W-I(p))),kt))')
    before='fac=rows(tv)/(rows(tv)-1)*(n-1)/(n-k)';assert before in text
    text=text.replace(before,'fac=rows(tv)/(rows(tv)-kt)')
    return text


def prepare_stata_visitor_rigour():
    for child in ['', 'data','do','native','native_tables','mcp']:(OUT/child).mkdir(parents=True,exist_ok=True)
    for branch in ['monthly','annual','ppml','diagnostics']:assert json.loads((ROOT/f'outputs/visitor_rigour/{branch}/verification.json').read_text())['status']=='passed'
    source_path=ROOT/'outputs/visitor_rigour/monthly/specifications.csv';source=pd.read_csv(source_path,dtype={'unit_id':str},float_precision='round_trip')
    rows=arrival_matrix();hashes={relative(source_path):sha(source_path)}
    for branch in ['monthly','annual','ppml','diagnostics']:
        path=ROOT/f'outputs/visitor_rigour/{branch}/verification.json';hashes[relative(path)]=sha(path)
    for path in [ROOT/'outputs/pyfixest/fit_log.csv',ROOT/'outputs/visitor_arrivals/pyfixest/fit_log.csv',ROOT/'outputs/visitor_rigour/monthly/pyfixest/fit_log.csv',ROOT/'notes/visitor_rigour_protocol.md',ROOT/'notes/crisis_sensitivity_policy.json']:
        hashes[relative(path)]=sha(path)
    datasets={}
    for order,row in enumerate(rows,1):
        target=source.loc[source.panel.eq(row['panel'])&source.outcome.eq(row['outcome'])&source.control_set.eq(row['control_set'])&source.sample_variant.eq(row['sample_variant'])&source.model.eq(row['structure'])&source.model_spec.eq('own_current')]
        assert len(target)==1;target=target.iloc[0];path=ROOT/target.sample_csv;hashes[relative(path)]=sha(path)
        frame=pd.read_csv(path,dtype={'unit_id':str},float_precision='round_trip').sort_values(['time','unit_id']).reset_index(drop=True)
        frame['sector_id']=frame.unit_id;frame['sector']=frame.unit;frame['period']=frame.month;frame['row_id']=range(1,len(frame)+1)
        columns=['sector_id','sector','period','time','year','season','row_id','arrivals','y_mom_nsa','y_yoy_nsa','temp_c','rain_100mm','rh_pct','lny_window_days','lny_lag1','wind_mps','pm25_10ug']
        frame=frame[columns]
        data_path=OUT/'data'/f"{row['panel']}__{row['outcome']}__{row['control_set']}__{row['sample_variant']}.csv"
        if str(data_path) not in datasets:frame.to_csv(data_path,index=False);datasets[str(data_path)]=sha(data_path)
        units=sorted(frame.sector_id.unique());labels=frame.drop_duplicates('sector_id').set_index('sector_id').sector.to_dict()
        weather=['temp_c','rain_100mm','rh_pct','lny_window_days','lny_lag1']+(['wind_mps','pm25_10ug'] if row['control_set']=='full' else [])
        aliases={f'x{i:03}':term for i,term in enumerate(weather)}
        if row['structure']=='H1':aliases.update({f'd{i:03}':f'temp_c:origin[{unit}]' for i,unit in enumerate(units[1:],1)})
        row.update(order=order,source_spec_id=target.spec_id,source_sample_csv=relative(path),source_sample_sha256=sha(path),source_design_npz=target.design_npz,
            source_design_sha256=sha(ROOT/target.design_npz),sample_csv=relative(data_path),sample_sha256=sha(data_path),n_obs=len(frame),n_periods=frame.time.nunique(),n_sectors=len(units),sector_ids=units,sector_names=labels,
            regressor_aliases=aliases,weather_terms=weather,horizon=12,growth_horizon=1 if row['outcome']=='y_mom_nsa' else 12,
            reference_sector_id=units[0],mean_y=float(frame[row['outcome']].mean()),sd_y=float(frame[row['outcome']].std()),
            formula=target.formula if not row['relative_only'] else 'Relative current temperature deviations; shared weather omitted by full actual-month FE + origin×calendar-month FE',
            primary_inference='Actual-calendar HAC12/24; T/(T−max local full-design rank), approximate t(T−1)' if not row['relative_only'] else 'Point identification diagnostic only; absolute national temperature absorbed; no SE/p claim')
    (OUT/'specifications.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    pd.DataFrame([{key:json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()} for row in rows]).to_csv(OUT/'specifications.csv',index=False)
    mata=arrivals_mata();(OUT/'do/postprocess.mata.do').write_text(mata)
    do=['version 17','clear all','set more off','set type double','set linesize 160',f'cd "{ROOT}"',f'log using "{OUT}/native/master_stata_output.txt", text replace','which reghdfe','which ftools','which esttab',f'do "{OUT}/do/postprocess.mata.do"',
        f'file open meta using "{OUT}/native/runs.csv", write replace','file write meta "target_id,run_variant,n_obs,n_eligible,p,df_m,df_a,df_a_nested,df_r,n_clusters,n_singletons,r_squared,converged,relative_only" _n']
    for row in rows:
        target=row['target_id'];folder=OUT/'native'/target;folder.mkdir(exist_ok=True)
        rhs=' '.join(row['regressor_aliases']);p=len(row['regressor_aliases']);g=row['n_sectors']
        do += [f'display "ARRIVAL TARGET {row["order"]}/38 {target}"',f'import delimited using "{ROOT/row["sample_csv"]}", clear varnames(1) stringcols(1) bindquote(strict) encoding("utf-8")','isid row_id','isid sector_id time','egen long industry_id=group(sector_id)',f'generate double y={row["outcome"]}']
        for i,term in enumerate(row['weather_terms']):do += [f'quietly summarize {term}',f'generate double x{i:03}={term}-r(mean)',f'label variable x{i:03} "{term}"']
        if row['structure']=='H1':
            for i,unit in enumerate(row['sector_ids'][1:],1):do += [f'generate double d{i:03}=x000*(industry_id=={i+1})',f'label variable d{i:03} "Temperature x {row["sector_names"][unit][:55]}"']
        absorb='industry_id#season time' if row['relative_only'] else 'industry_id#season year';vce='robust' if row['relative_only'] or g==1 else 'cluster industry_id'
        command=f'reghdfe y {rhs}, absorb({absorb}) vce({vce}) tolerance(1e-12) iterate(16000) residuals(__resid)'
        do += [command,f'estimates store a{row["order"]:03}','generate byte __sample=e(sample)',f'local needs_exact=(e(N)<{row["n_obs"]})']
        for run in ['native_default','exact_eligible']:
            dest=folder/run;dest.mkdir(exist_ok=True)
            if run=='exact_eligible':do += ['if `needs_exact\' {','drop __resid __sample',command+' keepsingletons','generate byte __sample=e(sample)',f'assert e(N)=={row["n_obs"]}']
            do += [f'export delimited row_id time industry_id __sample __resid using "{dest}/sample_residuals.csv", replace']
            if row['relative_only']:
                do += [f'mata: sk_write("{dest}/native_beta.csv",st_matrix("e(b)")[1,1..{p}]\')',f'mata: sk_write("{dest}/native_vcov.csv",st_matrix("e(V)")[1..{p},1..{p}])']
            else:do += [f'mata: sk_post("{dest}","{rhs}",12)']
            do += ['local ng=cond("`e(vce)\'"=="cluster",e(N_clust),0)',f'file write meta "{target},{run}," %21.17g (e(N)) ",{row["n_obs"]},{p}," %21.17g (e(df_m)) "," %21.17g (e(df_a)) "," %21.17g (e(df_a_nested)) "," %21.17g (e(df_r)) "," %21.17g (`ng\') "," %21.17g (e(num_singletons)) "," %21.17g (e(r2)) "," %21.17g (e(converged)) ",{int(row["relative_only"])}" _n']
            if run=='exact_eligible':do += ['}']
    do += ['file close meta']
    groups={}
    for row in rows:groups.setdefault((row['panel'],row['outcome'],row['control_set'],row['structure'],row['relative_only']),[]).append(row)
    for key,part in groups.items():
        file='__'.join(map(str,key))+'.html';stored=' '.join(f'a{r["order"]:03}' for r in part);titles=' '.join('"'+r['sample_variant']+'"' for r in part)
        display='cells(b(fmt(6))) nostar' if key[3]=='H1' else 'b(6) se(6) star(* 0.10 ** 0.05 *** 0.01)'
        note='Relative-only actual-date-FE coefficients; absolute temperature is absorbed; no inference claim.' if key[4] else 'Native H1 unit-cluster VCE can be highly singular: coefficients only; joint calendar main inference in validated CSV.' if key[3]=='H1' else 'Native default origin-cluster/robust convention; arrival main calendar correction is exported separately.'
        do += [f'esttab {stored} using "{OUT}/native_tables/{file}", replace html label {display} stats(N r2 df_a df_a_nested) mtitles({titles}) title("Native reghdfe arrivals: {" / ".join(map(str,key))}") addnotes("{note}")']
    do += [f'file open done using "{OUT}/native/completed.txt", write replace','file write done "38 arrival target analyses completed" _n','file close done','log close']
    (OUT/'do/key_results.do').write_text('\n'.join(do)+'\n')
    manifest={'status':'prepared_not_executed','targets':38,'regular_targets':36,'relative_date_FE_point_checks':2,
        'selection':'Prespecified current models;18longcommon+6fullcommonall+4fullcountryevent/strict+8H1currentall+2relative-date-FE diagnostics; no sign/significance selection',
        'inputs_sha256':hashes,'prepared_samples_sha256':{relative(path):digest for path,digest in datasets.items()},'do_sha256':sha(OUT/'do/key_results.do'),'mata_sha256':sha(OUT/'do/postprocess.mata.do'),
        'shared_mata_source':'scripts/stata_key_results.py','shared_mata_source_sha256':sha(ROOT/'scripts/stata_key_results.py'),
        'preserved_heat_verification_sha256':sha(ROOT/'outputs/stata_key_results/verification.json'),
        'calendar_rule':'Static current all/event/full[t−12,t] strict support from verified monthly own_current branch. No AR weather-history restriction borrowed.',
        'inference':'Native default VCE retained separately. Main regular effects: actual-calendar Bartlett12/24, T/(T−max local full-X rank), approximate t(T−1). H1 origin-cluster covariance is diagnostic only; relative-date-FE checks are point-only.'}
    assert all(sha(ROOT/path)==digest for path,digest in hashes.items())
    (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(f'Prepared38 native Stata arrival targets and{len(datasets)} datasets.',flush=True);return manifest


if __name__=='__main__':prepare_stata_visitor_rigour()
