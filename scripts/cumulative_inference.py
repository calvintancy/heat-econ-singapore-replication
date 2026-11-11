"""Sector-CR1 inference for a common-temperature linear contrast."""
import numpy as np
from scipy.stats import t
from sector_cluster import sector_cluster_covariance


def cluster_sum(model, sample, terms, key='sum'):
    covariance,meta=sector_cluster_covariance(model,sample.sector_id)
    positions=[model.model.exog_names.index(term) for term in terms]
    beta=float(model.params[terms].sum())
    se=float(np.sqrt(covariance[np.ix_(positions,positions)].sum()))
    critical=t.ppf(.975,meta['inference_df'])
    return {key:beta,'std_error':se,'ci_lower':beta-critical*se,'ci_upper':beta+critical*se,
            'p_value':float(2*t.sf(abs(beta/se),meta['inference_df'])),'covariance':'Sector cluster',**meta}
