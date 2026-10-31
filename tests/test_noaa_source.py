from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
from app.services.noaa_source import _cycle, _query, SOURCE_CONFIG

def test_noaa_cycle_rounds_down_to_synoptic_cycle():
    value=datetime(2026,10,29,14,45,tzinfo=timezone.utc)
    assert _cycle(value)==datetime(2026,10,29,12,0,tzinfo=timezone.utc)

def test_gfs_query_uses_requested_bbox_and_upper_air_fields():
    url=_query('GFS',datetime(2026,10,29,12,tzinfo=timezone.utc),3,-6.1,105.4,'upper')
    parsed=urlparse(url); params=parse_qs(parsed.query)
    assert parsed.netloc=='nomads.ncep.noaa.gov'
    assert params['file']==['gfs.t12z.pgrb2.0p25.f003']
    assert 'lev_1000_mb' in params and 'lev_50_mb' in params
    assert 'var_UGRD' in params and 'var_VGRD' in params

def test_gdas_only_analysis_cycle_is_configured():
    url=_query('GDAS',datetime(2026,10,29,0,tzinfo=timezone.utc),0,-6.1,105.4,'upper')
    assert 'filter_gdas_0p25.pl' in url
    assert SOURCE_CONFIG['GFS']['label']=='NOAA GFS 0.25°'
