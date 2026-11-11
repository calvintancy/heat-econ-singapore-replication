"""Verify the visible baseline is the already-estimated full-control specification."""
import json
from pathlib import Path
import pandas as pd
from lxml import html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'


def verify_baseline():
    doc=html.fromstring((OUT/'heat_economy.html').read_text(),parser=html.HTMLParser(huge_tree=True))
    script='\n'.join(doc.xpath('//script[not(@type)]/text()'))
    assert "window:'full'" in script
    buttons=doc.xpath('//button[@data-window and @aria-pressed="true"]')
    assert len(buttons)==1 and buttons[0].get('data-window')=='full'
    data=json.loads(doc.xpath('//script[@id="collection-data"]/text()')[0])
    baseline=[r for r in data['overview'] if r['is_baseline']]
    assert len(baseline)==9*3 and all(r['window']=='full' for r in baseline)
    for row in baseline:
        source=pd.read_csv(ROOT/row['source_csv']);model=source[source.model.eq(row['model'])]
        assert len(model)>0
        for term in ['rain_100mm','rh_pct','wind_mps','pm25_10ug']:
            assert model.formula.str.contains(term,regex=False).all(),(row['source_csv'],row['model'],term)
    for name in ['granular_va.html','manufacturing_detail.html']:
        chapter=html.fromstring((OUT/name).read_text())
        visible=chapter.xpath('//section[contains(concat(" ",normalize-space(@class)," ")," result-view ") and not(@hidden)]')
        assert len(visible)==1 and visible[0].get('data-window')=='full'
        select=chapter.xpath('//select[@id="window-select"]/option')[0]
        assert select.get('value')=='full'
        if name=='manufacturing_detail.html':assert visible[0].get('data-context')=='iip-monthly-detailed'
        tables=visible[0].xpath('.//div[contains(@class,"interaction-native")]//table')
        assert tables and 'PM2.5' in tables[0].text_content()
    broad=html.fromstring((OUT/'sector_heterogeneity.html').read_text())
    assert broad.xpath('//button[@data-kind="controls" and @data-value="full" and @aria-pressed="true"]')
    assert "controls:'full'" in '\n'.join(broad.xpath('//script/text()'))
    growth=html.fromstring((OUT/'outcome_comparisons.html').read_text())
    assert growth.xpath('//div[@id="va-full" and not(@hidden)]')
    assert not growth.xpath('//div[@id="va-weather" and not(@hidden)]')
    aggregate=html.fromstring((OUT/'aggregate_va.html').read_text())
    text=aggregate.text_content()
    assert text.index('Baseline · rainfall, humidity, wind and PM2.5')<text.index('Robustness · baseline controls, observations dated 2020–2022 excluded')
    result={'status':'passed','baseline_overview_rows':len(baseline),'controls':['rainfall','humidity','wind','PM2.5'],
        'window':'All eligible 2017–2025 dates','default_views_checked':['overview','detailed VA','monthly detailed IIP','broad heterogeneity','growth/levels','aggregate VA'],
        'exclusion':'COVID exclusion and no-PM long history are robustness, not baseline'}
    target=OUT/'collection/baseline_verification.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified full-control PM2.5 baseline defaults, formulas and visible main FE tables.')
    return result

if __name__=='__main__':verify_baseline()
