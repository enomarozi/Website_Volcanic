from pathlib import Path
import math
import numpy as np
from netCDF4 import Dataset,num2date

class MeteorologyService:
    def __init__(self,file_path:str):
        self.file_path=Path(file_path)
        self.dataset=None
        self.data=None

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
        time_values=np.asarray(time_variable[:])

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

        temperature_variable=variables["temperature"] if "temperature" in variables else variables["t"] if "t" in variables else None
        geopotential_variable=variables["geopotential"] if "geopotential" in variables else variables["z"] if "z" in variables else None

        self.data={
            "latitude":np.asarray(variables["latitude"][:],dtype=float),
            "longitude":np.asarray(variables["longitude"][:],dtype=float),
            "pressure_level":np.asarray(variables["pressure_level"][:],dtype=float),
            "time":time_values,
            "times":times,
            "u":np.asarray(variables["u"][:],dtype=float),
            "v":np.asarray(variables["v"][:],dtype=float),
            "temperature":np.asarray(temperature_variable[:],dtype=float) if temperature_variable is not None else None,
            "geopotential":np.asarray(geopotential_variable[:],dtype=float) if geopotential_variable is not None else None
        }

        self._validate()
        return self.data

    def _validate(self):
        data=self.data

        if data["u"].ndim!=4 or data["v"].ndim!=4:
            raise ValueError("U and V wind data must have dimensions (time, pressure, latitude, longitude).")

        if data["u"].shape!=data["v"].shape:
            raise ValueError("U and V wind dimensions do not match.")

        if len(data["times"])!=data["u"].shape[0]:
            raise ValueError("Meteorological time dimension does not match wind data.")

        if len(data["pressure_level"])!=data["u"].shape[1]:
            raise ValueError("Pressure level dimension does not match wind data.")

        if len(data["latitude"])!=data["u"].shape[2]:
            raise ValueError("Latitude dimension does not match wind data.")

        if len(data["longitude"])!=data["u"].shape[3]:
            raise ValueError("Longitude dimension does not match wind data.")

        if data["temperature"] is not None and data["temperature"].shape!=data["u"].shape:
            raise ValueError("Temperature dimensions do not match wind data.")

        if data["geopotential"] is not None and data["geopotential"].shape!=data["u"].shape:
            raise ValueError("Geopotential dimensions do not match wind data.")

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
        times=info["times"]
        elapsed=info["elapsed_seconds"]

        if not times:
            raise ValueError("Meteorological dataset contains no valid time.")

        target=max(0.0,float(elapsed_seconds))

        if not elapsed:
            return 0

        return min(
            range(len(elapsed)),
            key=lambda index:abs(elapsed[index]-target)
        )
    @staticmethod
    def _axis_spacing(values):
        values=np.asarray(values,dtype=float)

        if values.size<2:
            return None

        differences=np.abs(np.diff(values))
        differences=differences[differences>0]

        return float(np.median(differences)) if differences.size else None

    def horizontal_resolution(self,latitude=None):
        data=self.load()

        lat=data["latitude"]
        lon=data["longitude"]

        lat_step=self._axis_spacing(lat)
        lon_step=self._axis_spacing(lon)

        if lat_step is None and lon_step is None:
            return 1000.0

        latitude=float(latitude if latitude is not None else np.mean(lat))

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

        lat_radius=min(
            abs(latitude-lat_min),
            abs(lat_max-latitude)
        )*111320.0

        lon_radius=min(
            abs(longitude-lon_min),
            abs(lon_max-longitude)
        )*111320.0*cos_lat

        radius=min(lat_radius,lon_radius)

        if radius<=0:
            radius=max(lat_radius,lon_radius)

        return float(max(radius,1000.0))

    def _precipitation_variable(self):
        data=self.load()

        names=[
            "tp",
            "total_precipitation",
            "precipitation",
            "precip",
            "pr",
            "rain",
            "rainfall",
            "precipitation_amount"
        ]

        variables=self.dataset.variables if self.dataset is not None else {}

        for name in names:
            if name in variables:
                return name

        for name,variable in variables.items():
            text=f"{name} {getattr(variable,'long_name','')} {getattr(variable,'standard_name','')}".lower()

            if "precip" in text or "rainfall" in text:
                return name

        return None

    def precipitation(self,time_index=0,latitude=None,longitude=None):
        self.load()

        name=self._precipitation_variable()

        if not name:
            return 0.0

        variable=self.dataset.variables[name]
        values=np.asarray(variable[:],dtype=float)

        if values.ndim==0:
            return max(0.0,float(values))

        time_index=max(
            0,
            min(int(time_index),len(self.data["times"])-1)
        )

        if values.ndim==1:
            return max(
                0.0,
                float(values[min(time_index,values.shape[0]-1)])
            )

        if values.ndim>=3:
            latitudes=self.data["latitude"]
            longitudes=self.data["longitude"]

            lat_index=int(
                np.abs(
                    latitudes-float(
                        latitude if latitude is not None else np.mean(latitudes)
                    )
                ).argmin()
            )

            lon_index=int(
                np.abs(
                    longitudes-float(
                        longitude if longitude is not None else np.mean(longitudes)
                    )
                ).argmin()
            )

            return max(
                0.0,
                float(values[time_index,lat_index,lon_index])
            )

        return max(0.0,float(values[time_index]))

    def nearest_grid(self,latitude,longitude):
        data=self.load()

        lat_index=int(
            np.abs(data["latitude"]-float(latitude)).argmin()
        )

        lon_index=int(
            np.abs(data["longitude"]-float(longitude)).argmin()
        )

        return {
            "latitude":float(data["latitude"][lat_index]),
            "longitude":float(data["longitude"][lon_index]),
            "latitude_index":lat_index,
            "longitude_index":lon_index
        }

    def atmospheric_profile(self,latitude,longitude,time_index=0):
        data=self.load()

        time_index=max(
            0,
            min(int(time_index),len(data["times"])-1)
        )

        grid=self.nearest_grid(latitude,longitude)

        lat_index=grid["latitude_index"]
        lon_index=grid["longitude_index"]

        pressure=np.asarray(
            data["pressure_level"],
            dtype=float
        )

        u=np.asarray(
            data["u"][time_index,:,lat_index,lon_index],
            dtype=float
        )

        v=np.asarray(
            data["v"][time_index,:,lat_index,lon_index],
            dtype=float
        )

        temperature=np.asarray(
            data["temperature"][time_index,:,lat_index,lon_index],
            dtype=float
        ) if data["temperature"] is not None else None

        geopotential=np.asarray(
            data["geopotential"][time_index,:,lat_index,lon_index],
            dtype=float
        ) if data["geopotential"] is not None else None

        altitude=(
            geopotential/9.80665
            if geopotential is not None
            else None
        )

        potential_temperature=None

        if temperature is not None:
            pressure_safe=np.maximum(pressure,0.001)

            potential_temperature=(
                temperature*
                (1000.0/pressure_safe)**(287.05/1004.67)
            )

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
                level["potential_temperature"]=float(
                    potential_temperature[index]
                )

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

        valid=[]

        for level in levels:
            try:
                z=float(level["altitude"])
                u=float(level["u"])
                v=float(level["v"])
                theta=level.get("potential_temperature")

                if not all(math.isfinite(value) for value in (z,u,v)):
                    continue

                theta=float(theta) if theta is not None else None
                if theta is not None and not math.isfinite(theta):
                    theta=None

                valid.append((z,u,v,theta))
            except (KeyError,TypeError,ValueError):
                continue

        if not valid:
            raise ValueError("Atmospheric profile contains no valid wind levels.")

        valid.sort(key=lambda item:item[0])
        target=float(altitude)

        if target<=valid[0][0]:
            z1,u,v,theta=valid[0]
        elif target>=valid[-1][0]:
            z1,u,v,theta=valid[-1]
        else:
            z1=u1=v1=theta1=None

            for lower,upper in zip(valid,valid[1:]):
                z_lower,u_lower,v_lower,t_lower=lower
                z_upper,u_upper,v_upper,t_upper=upper

                if z_lower<=target<=z_upper:
                    fraction=(target-z_lower)/(z_upper-z_lower) if z_upper!=z_lower else 0.0
                    u=u_lower+(u_upper-u_lower)*fraction
                    v=v_lower+(v_upper-v_lower)*fraction
                    theta=(
                        t_lower+(t_upper-t_lower)*fraction
                        if t_lower is not None and t_upper is not None
                        else None
                    )
                    break

            if u is None or v is None:
                z1,u,v,theta=valid[0]

        wind_speed=math.hypot(u,v)

        theta_values=[
            item[3]
            for item in valid
            if item[3] is not None and math.isfinite(item[3])
        ]

        if len(theta_values)>=2:
            theta_mean=sum(theta_values)/len(theta_values)
            z_bottom=valid[0][0]
            z_top=valid[-1][0]
            theta_bottom=valid[0][3]
            theta_top=valid[-1][3]

            if (
                theta_bottom is not None
                and theta_top is not None
                and z_top>z_bottom
            ):
                dtheta_dz=(theta_top-theta_bottom)/(z_top-z_bottom)
                brunt_vaisala=math.sqrt(
                    max(
                        9.81/max(theta_mean,1e-12)*
                        max(dtheta_dz,0.0),
                        0.0
                    )
                )
            else:
                brunt_vaisala=0.011
        else:
            brunt_vaisala=0.011

        brunt_vaisala=max(float(brunt_vaisala),0.011)

        return {
            "u":float(u),
            "v":float(v),
            "wind_speed":float(wind_speed),
            "brunt_vaisala":float(brunt_vaisala),
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