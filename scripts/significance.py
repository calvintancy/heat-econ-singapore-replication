"""Explicit significance categories computed from saved, unrounded p-values."""
import math


def _valid(p):
    try:
        value=float(p)
    except (TypeError,ValueError): return None
    return value if math.isfinite(value) and 0<=value<=1 else None


def significance_label(p):
    p=_valid(p)
    if p is None: return 'Unavailable'
    if p<.01: return 'Significant at 1%'
    if p<.05: return 'Significant at 5%'
    if p<.10: return '10% only'
    return 'Not significant at 10%'


def significance_stars(p):
    p=_valid(p)
    return '' if p is None else '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''


def significance_class(p):
    p=_valid(p)
    return 'unavailable' if p is None else 'sig-1' if p<.01 else 'sig-5' if p<.05 else 'sig-10' if p<.10 else 'not-sig'


def p_format(p):
    p=_valid(p)
    if p is None: return 'Unavailable'
    if p<.001: return '<0.001'
    if p not in [.01,.05,.10] and round(p,3) in [.01,.05,.10]: return f'{p:.5f}'
    return f'{p:.3f}'
