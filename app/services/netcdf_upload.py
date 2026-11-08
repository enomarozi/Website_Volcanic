from pathlib import Path
from datetime import datetime
import json
import shutil
import numpy as np
from netCDF4 import Dataset,num2date

REQUIRED_UPPER={"latitude","longitude","pressure_level","valid_time","u","v"}
SURFACE_VARIABLES={"t2m","zust","sshf","tp","precipitation","precip","rain","total_precipitation"}

def _times(ds):
    if "valid_time" not in ds.variables:
        return []
    var=ds.variables["valid_time"]
    values=np.asarray(var[:])
    if hasattr(var,"units"):
        try:
            dates=num2date(values,units=var.units,calendar=getattr(var,"calendar","standard"),only_use_cftime_datetimes=False,only_use_python_datetimes=True)
            return list(dates)
        except Exception:
            pass
    return [v.item() if hasattr(v,"item") else v for v in values]

def inspect_netcdf(path:Path,kind:str):
    with Dataset(str(path),"r") as ds:
        variables=set(ds.variables.keys())
        dimensions={k:int(len(v)) for k,v in ds.dimensions.items()}
        if kind=="upper_air":
            missing=REQUIRED_UPPER-variables
            if missing:
                raise ValueError(f"Upper-air missing variables: {', '.join(sorted(missing))}")
            u=ds.variables["u"]
            v=ds.variables["v"]
            if u.ndim!=4 or v.ndim!=4:
                raise ValueError("u dan v harus memiliki 4 dimensi: time, pressure, latitude, longitude")
            times=_times(ds)
            if not times:
                raise ValueError("valid_time tidak dapat dibaca")
            pressure=np.asarray(ds.variables["pressure_level"][:],dtype=float)
            lat=np.asarray(ds.variables["latitude"][:],dtype=float)
            lon=np.asarray(ds.variables["longitude"][:],dtype=float)
            if lat.size==0 or lon.size==0 or pressure.size==0:
                raise ValueError("Coordinate NetCDF kosong")
            return {
                "variables":sorted(variables),
                "dimensions":dimensions,
                "time_count":len(times),
                "start_time":str(times[0]),
                "end_time":str(times[-1]),
                "interval_hours":_interval_hours(times),
                "latitude_min":float(np.min(lat)),
                "latitude_max":float(np.max(lat)),
                "longitude_min":float(np.min(lon)),
                "longitude_max":float(np.max(lon)),
            }
        if kind=="surface":
            if "valid_time" not in variables:
                raise ValueError("Surface dataset tidak memiliki valid_time")
            available=variables&SURFACE_VARIABLES
            if not available:
                raise ValueError("Surface dataset tidak memiliki variabel surface yang didukung")
            times=_times(ds)
            if not times:
                raise ValueError("valid_time tidak dapat dibaca")
            return {
                "variables":sorted(available),
                "dimensions":dimensions,
                "time_count":len(times),
                "start_time":str(times[0]),
                "end_time":str(times[-1]),
                "interval_hours":_interval_hours(times)
            }
    raise ValueError("Dataset type tidak valid")

def _interval_hours(times):
    if len(times)<2:
        return None
    try:
        values=[(times[i+1]-times[i]).total_seconds()/3600 for i in range(len(times)-1)]
        return float(np.median(values))
    except Exception:
        return None

def save_upload(upload_file,destination:Path):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file,buffer)
    return destination

def unique_filename(name:str,prefix:str):
    stem=Path(name).stem
    stamp=datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{stamp}_{stem}.nc"