from pathlib import Path
import math
import numpy as np
from netCDF4 import Dataset, num2date

class MeteorologyService:
    def __init__(self,file_path:str):
        self.file_path=Path(file_path)
        self.dataset=None
        self.data=None

    @staticmethod
    def _array(variable):
        values=np.ma.asarray(variable[:],dtype=float)
        return np.asarray(values.filled(np.nan),dtype=float)

    @staticmethod
    def _valid(value):
        try:
            value=float(value)
            return value if math.isfinite(value) else None
        except (TypeError,ValueError):
            return None

    def load(self):
        if self.data is not None:
            return self.data
        if not self.file_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {self.file_path}")

        self.dataset=Dataset(self.file_path)
        variables=self.dataset.variables
        required=("latitude","longitude","pressure_level","valid_time","u","v")
        missing=[name for name in required if name not in variables]
        if missing:
            self.close()
            raise ValueError(f"Missing required NetCDF variables: {', '.join(missing)}")

        time_variable=variables["valid_time"]
        time_values=self._array(time_variable)

        try:
            times=list(num2date(
                time_values,
                units=time_variable.units,
                calendar=getattr(time_variable,"calendar","standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True
            ))
        except Exception:
            times=[value.item() if hasattr(value,"item") else value for value in time_values]

        temperature_variable=variables.get("temperature",variables.get("t"))
        geopotential_variable=variables.get("geopotential",variables.get("z"))

        self.data={
            "latitude":self._array(variables["latitude"]),
            "longitude":self._array(variables["longitude"]),
            "pressure_level":self._array(variables["pressure_level"]),
            "time":time_values,
            "times":times,
            "u":self._array(variables["u"]),
            "v":self._array(variables["v"]),
            "temperature":self._array(temperature_variable) if temperature_variable is not None else None,
            "geopotential":self._array(geopotential_variable) if geopotential_variable is not None else None
        }

        try:
            self._validate()
        except Exception:
            self.close()
            raise

        return self.data

    def _validate(self):
        data=self.data
        u=data["u"]
        v=data["v"]

        if u.ndim!=4 or v.ndim!=4:
            raise ValueError("U and V wind data must have dimensions (time, pressure, latitude, longitude).")
        if u.shape!=v.shape:
            raise ValueError("U and V wind dimensions do not match.")
        if len(data["times"])!=u.shape[0]:
            raise ValueError("Meteorological time dimension does not match wind data.")
        if len(data["pressure_level"])!=u.shape[1]:
            raise ValueError("Pressure level dimension does not match wind data.")
        if len(data["latitude"])!=u.shape[2]:
            raise ValueError("Latitude dimension does not match wind data.")
        if len(data["longitude"])!=u.shape[3]:
            raise ValueError("Longitude dimension does not match wind data.")
        for name in ("temperature","geopotential"):
            values=data[name]
            if values is not None and values.shape!=u.shape:
                raise ValueError(f"{name.capitalize()} dimensions do not match wind data.")
        if not len(data["times"]) or not len(data["pressure_level"]):
            raise ValueError("Meteorological dataset contains no time or pressure levels.")

    def time_information(self):
        data=self.load()
        times=data["times"]
        elapsed=[]

        if times:
            first=times[0]
            for index,current in enumerate(times):
                try:
                    elapsed.append(float((current-first).total_seconds()))
                except Exception:
                    elapsed.append(float(index*3600.0))

        interval_seconds=0.0
        if len(times)>1:
            try:
                interval_seconds=max(0.0,float((times[1]-times[0]).total_seconds()))
            except Exception:
                interval_seconds=3600.0

        return {
            "count":len(times),
            "times":times,
            "elapsed_seconds":elapsed,
            "interval_seconds":interval_seconds,
            "interval_hours":interval_seconds/3600.0 if interval_seconds>0 else None
        }

    def time_index_at_elapsed(self,elapsed_seconds):
        info=self.time_information()
        elapsed=info["elapsed_seconds"]
        if not info["times"]:
            raise ValueError("Meteorological dataset contains no valid time.")
        target=max(0.0,float(elapsed_seconds))
        return min(range(len(elapsed)),key=lambda index:abs(elapsed[index]-target)) if elapsed else 0

    @staticmethod
    def _axis_spacing(values):
        values=np.asarray(values,dtype=float)
        if values.size<2:
            return None
        differences=np.abs(np.diff(values))
        differences=differences[np.isfinite(differences)&(differences>0)]
        return float(np.median(differences)) if differences.size else None

    def horizontal_resolution(self,latitude=None):
        data=self.load()
        lat_step=self._axis_spacing(data["latitude"])
        lon_step=self._axis_spacing(data["longitude"])
        if lat_step is None and lon_step is None:
            return 1000.0
        latitude=float(latitude if latitude is not None else np.mean(data["latitude"]))
        lat_m=(lat_step or lon_step)*111320.0
        lon_m=(lon_step or lat_step)*111320.0*max(abs(math.cos(math.radians(latitude))),1e-8)
        values=[value for value in (lat_m,lon_m) if value>0]
        return float(min(values)) if values else 1000.0

    def domain_radius(self,latitude=None,longitude=None):
        data=self.load()
        latitude=float(latitude if latitude is not None else np.mean(data["latitude"]))
        longitude=float(longitude if longitude is not None else np.mean(data["longitude"]))
        cos_lat=max(abs(math.cos(math.radians(latitude))),1e-8)
        lat_min=float(np.min(data["latitude"]))
        lat_max=float(np.max(data["latitude"]))
        lon_min=float(np.min(data["longitude"]))
        lon_max=float(np.max(data["longitude"]))
        lat_radius=min(abs(latitude-lat_min),abs(lat_max-latitude))*111320.0
        lon_radius=min(abs(longitude-lon_min),abs(lon_max-longitude))*111320.0*cos_lat
        radius=min(lat_radius,lon_radius)
        if radius<=0:
            radius=max(lat_radius,lon_radius)
        return float(max(radius,1000.0))

    def _precipitation_variable(self):
        self.load()
        variables=self.dataset.variables
        names=("tp","total_precipitation","precipitation","precip","pr","rain","rainfall","precipitation_amount")

        for name in names:
            if name in variables:
                return name

        for name,variable in variables.items():
            description=f"{name} {getattr(variable,'long_name','')} {getattr(variable,'standard_name','')}".lower()
            if "precip" in description or "rainfall" in description:
                return name
        return None

    def _precipitation_to_meters(self,value,variable):
        units=str(getattr(variable,"units","")).lower().replace(" ","").replace("**","")
        if units in ("kgm-2","kg/m2","kgm^-2","mm","millimeter","millimeters","millimetre","millimetres"):
            return value/1000.0
        if units in ("m","meter","meters","metre","metres"):
            return value
        return value

    def precipitation(self,time_index=0,latitude=None,longitude=None):
        self.load()
        name=self._precipitation_variable()
        if not name:
            return None

        variable=self.dataset.variables[name]
        values=self._array(variable)
        if values.size==0:
            return None

        dimensions=list(variable.dimensions)
        indexes={}

        for axis,dimension in enumerate(dimensions):
            key=dimension.lower()
            if key in ("valid_time","time","forecast_time"):
                indexes[axis]=max(0,min(int(time_index),values.shape[axis]-1))
            elif key in ("latitude","lat"):
                target=float(latitude if latitude is not None else np.mean(self.data["latitude"]))
                indexes[axis]=int(np.abs(self.data["latitude"]-target).argmin())
            elif key in ("longitude","lon"):
                target=float(longitude if longitude is not None else np.mean(self.data["longitude"]))
                indexes[axis]=int(np.abs(self.data["longitude"]-target).argmin())
            elif key in ("pressure_level","isobaricinhpa","level"):
                indexes[axis]=0

        if dimensions and len(indexes)==values.ndim:
            index=tuple(indexes[axis] for axis in range(values.ndim))
            result=self._valid(values[index])
        elif values.ndim==0:
            result=self._valid(values.item())
        elif values.ndim==1:
            result=self._valid(values[min(max(int(time_index),0),values.shape[0]-1)])
        else:
            return None

        if result is None:
            return None
        return max(0.0,float(self._precipitation_to_meters(result,variable)))

    def nearest_grid(self,latitude,longitude):
        data=self.load()
        lat_index=int(np.abs(data["latitude"]-float(latitude)).argmin())
        lon_index=int(np.abs(data["longitude"]-float(longitude)).argmin())
        return {
            "latitude":float(data["latitude"][lat_index]),
            "longitude":float(data["longitude"][lon_index]),
            "latitude_index":lat_index,
            "longitude_index":lon_index
        }

    def atmospheric_profile(self,latitude,longitude,time_index=0):
        data=self.load()
        time_index=max(0,min(int(time_index),len(data["times"])-1))
        grid=self.nearest_grid(latitude,longitude)
        lat_index=grid["latitude_index"]
        lon_index=grid["longitude_index"]

        pressure=data["pressure_level"]
        u=data["u"][time_index,:,lat_index,lon_index]
        v=data["v"][time_index,:,lat_index,lon_index]
        temperature=data["temperature"][time_index,:,lat_index,lon_index] if data["temperature"] is not None else None
        geopotential=data["geopotential"][time_index,:,lat_index,lon_index] if data["geopotential"] is not None else None
        altitude=geopotential.copy() if geopotential is not None else None

        potential_temperature=None
        if temperature is not None:
            pressure_safe=np.maximum(pressure,0.001)
            potential_temperature=temperature*(1000.0/pressure_safe)**(287.05/1004.67)

        profile=[]
        for index in range(len(pressure)):
            level={
                "pressure":float(pressure[index]),
                "u":float(u[index]),
                "v":float(v[index])
            }
            if temperature is not None:
                level["temperature"]=float(temperature[index])
            if geopotential is not None:
                level["geopotential"]=float(geopotential[index])
            if altitude is not None:
                level["altitude"]=float(altitude[index])
            if potential_temperature is not None:
                level["potential_temperature"]=float(potential_temperature[index])
            profile.append(level)

        return {
            "profile":profile,
            "pressure":pressure.tolist(),
            "u":u.tolist(),
            "v":v.tolist(),
            "temperature":temperature.tolist() if temperature is not None else None,
            "geopotential":geopotential.tolist() if geopotential is not None else None,
            "altitude":altitude.tolist() if altitude is not None else None,
            "potential_temperature":potential_temperature.tolist() if potential_temperature is not None else None
        }

    def wind_at_altitude(self,latitude,longitude,altitude,time_index=0):
        profile=self.atmospheric_profile(latitude,longitude,time_index)
        levels=profile.get("profile",[])

        if not levels:
            raise ValueError("Atmospheric profile contains no levels.")
        if profile.get("altitude") is None:
            raise ValueError("Geopotential data is required to calculate wind at altitude.")

        valid=[]
        for level in levels:
            try:
                z=self._valid(level.get("altitude"))
                u=self._valid(level.get("u"))
                v=self._valid(level.get("v"))
                theta=self._valid(level.get("potential_temperature"))
                if z is not None and u is not None and v is not None:
                    valid.append((z,u,v,theta))
            except (TypeError,ValueError):
                continue

        if not valid:
            raise ValueError("Atmospheric profile contains no valid wind levels.")

        valid.sort(key=lambda item:item[0])
        target=float(altitude)

        if target<=valid[0][0]:
            _,u,v,theta=valid[0]
        elif target>=valid[-1][0]:
            _,u,v,theta=valid[-1]
        else:
            lower=upper=None
            for first,second in zip(valid,valid[1:]):
                if first[0]<=target<=second[0]:
                    lower,upper=first,second
                    break

            if lower is None:
                _,u,v,theta=valid[0]
            else:
                z1,u1,v1,t1=lower
                z2,u2,v2,t2=upper
                fraction=(target-z1)/(z2-z1) if z2!=z1 else 0.0
                u=u1+(u2-u1)*fraction
                v=v1+(v2-v1)*fraction
                theta=t1+(t2-t1)*fraction if t1 is not None and t2 is not None else None

        wind_speed=math.hypot(u,v)
        theta_values=[item for item in valid if item[3] is not None]

        if len(theta_values)>=2:
            theta_mean=float(np.mean([item[3] for item in theta_values]))
            bottom=theta_values[0]
            top=theta_values[-1]
            if top[0]>bottom[0] and theta_mean>0:
                dtheta_dz=(top[3]-bottom[3])/(top[0]-bottom[0])
                brunt_vaisala=math.sqrt(max(9.81/theta_mean*max(dtheta_dz,0.0),0.0))
            else:
                brunt_vaisala=0.011
        else:
            brunt_vaisala=0.011

        return {
            "u":float(u),
            "v":float(v),
            "wind_speed":float(wind_speed),
            "brunt_vaisala":float(max(brunt_vaisala,0.011)),
            "altitude":float(target),
            "time_index":int(time_index)
        }

    def dataset_information(self):
        data=self.load()
        info=self.time_information()
        return {
            "file":str(self.file_path),
            "latitude_count":len(data["latitude"]),
            "longitude_count":len(data["longitude"]),
            "pressure_level_count":len(data["pressure_level"]),
            "time_count":len(data["times"]),
            "latitude_min":float(np.min(data["latitude"])),
            "latitude_max":float(np.max(data["latitude"])),
            "longitude_min":float(np.min(data["longitude"])),
            "longitude_max":float(np.max(data["longitude"])),
            "pressure_min":float(np.min(data["pressure_level"])),
            "pressure_max":float(np.max(data["pressure_level"])),
            "interval_hours":info["interval_hours"]
        }

    def close(self):
        if self.dataset is not None:
            try:
                self.dataset.close()
            finally:
                self.dataset=None
        self.data=None