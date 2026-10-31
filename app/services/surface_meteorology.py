from pathlib import Path
import numpy as np
from netCDF4 import Dataset,num2date

class SurfaceMeteorologyService:
    def __init__(self,instant_file:str,accum_file:str):
        self.instant_file=Path(instant_file)
        self.accum_file=Path(accum_file)
        self.data=None

    @staticmethod
    def _times(variable):
        values=np.asarray(variable[:])
        if hasattr(variable,"units"):
            try:
                return list(num2date(values,units=variable.units,calendar=getattr(variable,"calendar","standard"),only_use_cftime_datetimes=False,only_use_python_datetimes=True))
            except Exception:
                pass
        return [value.item() if hasattr(value,"item") else value for value in values]

    def load(self):
        if self.data is not None:
            return self.data
        if not self.instant_file.exists():
            raise FileNotFoundError(f"Instant NetCDF file not found: {self.instant_file}")
        if not self.accum_file.exists():
            raise FileNotFoundError(f"Accumulated NetCDF file not found: {self.accum_file}")

        with Dataset(self.instant_file) as instant,Dataset(self.accum_file) as accum:
            if "valid_time" not in instant.variables or "valid_time" not in accum.variables:
                raise ValueError("Surface NetCDF tidak memiliki valid_time.")
            required_instant=("t2m","sp","zust")
            required_accum=("tp","sshf")
            for name in required_instant:
                if name not in instant.variables:
                    raise ValueError(f"Variable {name} tidak ditemukan pada Surface Instant NetCDF.")
            for name in required_accum:
                if name not in accum.variables:
                    raise ValueError(f"Variable {name} tidak ditemukan pada Surface Accumulated NetCDF.")

            instant_times=self._times(instant.variables["valid_time"])
            accum_times=self._times(accum.variables["valid_time"])

            if len(instant_times)!=len(accum_times):
                raise ValueError("Jumlah waktu Surface Instant dan Accumulated berbeda.")

            self.data={
                "time":np.asarray(instant.variables["valid_time"][:]),
                "times":instant_times,
                "latitude":np.asarray(instant.variables["latitude"][:],dtype=float),
                "longitude":np.asarray(instant.variables["longitude"][:],dtype=float),
                "friction_velocity":np.asarray(instant.variables["zust"][:],dtype=float),
                "temperature":np.asarray(instant.variables["t2m"][:],dtype=float),
                "surface_pressure":np.asarray(instant.variables["sp"][:],dtype=float),
                "precipitation":np.asarray(accum.variables["tp"][:],dtype=float),
                "heat_flux_accumulated":np.asarray(accum.variables["sshf"][:],dtype=float),
                "precipitation_units":getattr(accum.variables["tp"],"units","m"),
                "heat_flux_units":getattr(accum.variables["sshf"],"units","J m**-2")
            }

        self._validate()
        return self.data

    def _validate(self):
        data=self.data
        expected=(len(data["times"]),len(data["latitude"]),len(data["longitude"]))
        for name in ("friction_velocity","temperature","surface_pressure","precipitation","heat_flux_accumulated"):
            if data[name].shape!=expected:
                raise ValueError(f"Unexpected shape for {name}. Expected {expected}, received {data[name].shape}")

        if data["precipitation_units"] not in ("m","meter","metre"):
            raise ValueError(f"Unsupported precipitation unit: {data['precipitation_units']}")

    @staticmethod
    def interval_seconds(times,time_index=0):
        if len(times)<2:
            return 0.0
        index=max(0,min(int(time_index),len(times)-2))
        try:
            return max(0.0,float((times[index+1]-times[index]).total_seconds()))
        except Exception:
            return 0.0

    def accumulated_flux_to_wm2(self,value,interval_seconds):
        interval_seconds=float(interval_seconds)
        if interval_seconds<=0.0:
            return 0.0
        return float(value)/interval_seconds

    @staticmethod
    def precipitation_m_to_mm(value):
        return max(float(value),0.0)*1000.0

    def time_index_at_elapsed(self,elapsed_seconds):
        data=self.load()
        if not data["times"]:
            return 0
        elapsed=[]
        first=data["times"][0]
        for index,current in enumerate(data["times"]):
            try:
                elapsed.append((current-first).total_seconds())
            except Exception:
                elapsed.append(index*3600.0)
        return int(np.abs(np.asarray(elapsed,dtype=float)-float(elapsed_seconds)).argmin())

    def surface_profile(self,latitude,longitude,time_index=0):
        data=self.load()
        lat_index=int(np.abs(data["latitude"]-float(latitude)).argmin())
        lon_index=int(np.abs(data["longitude"]-float(longitude)).argmin())
        time_index=max(0,min(int(time_index),len(data["times"])-1))
        interval=self.interval_seconds(data["times"],time_index)
        tp=max(float(data["precipitation"][time_index,lat_index,lon_index]),0.0)
        sshf=float(data["heat_flux_accumulated"][time_index,lat_index,lon_index])

        return {
            "latitude":float(data["latitude"][lat_index]),
            "longitude":float(data["longitude"][lon_index]),
            "time_index":int(time_index),
            "time":str(data["times"][time_index]),
            "interval_seconds":float(interval),
            "interval_hours":float(interval/3600.0) if interval>0.0 else None,
            "friction_velocity":float(data["friction_velocity"][time_index,lat_index,lon_index]),
            "temperature":float(data["temperature"][time_index,lat_index,lon_index]),
            "surface_pressure":float(data["surface_pressure"][time_index,lat_index,lon_index]),
            "precipitation":tp,
            "precipitation_m":tp,
            "precipitation_mm":self.precipitation_m_to_mm(tp),
            "precipitation_units":data["precipitation_units"],
            "heat_flux_accumulated":sshf,
            "heat_flux_wm2":self.accumulated_flux_to_wm2(sshf,interval),
            "heat_flux_units":data["heat_flux_units"]
        }

    def close(self):
        self.data=None
