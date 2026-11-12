
from pathlib import Path
import numpy as np
from netCDF4 import Dataset, num2date


class SurfaceMeteorologyService:
    def __init__(self, instant_file: str, accum_file: str):
        self.instant_file = Path(instant_file)
        self.accum_file = Path(accum_file)
        self.data = None

    @staticmethod
    def _times(variable):
        values = np.asarray(variable[...])
        if hasattr(variable, "units"):
            try:
                result = num2date(
                    values,
                    units=variable.units,
                    calendar=getattr(variable, "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                return list(np.atleast_1d(result))
            except Exception:
                pass
        return list(np.atleast_1d(values))

    @staticmethod
    def _seconds_between(start, end):
        try:
            return float((end-start).total_seconds())
        except (AttributeError, TypeError):
            return float((end-start)/np.timedelta64(1,"s"))

    @staticmethod
    def _normalize_units(units):
        return str(units or "").strip().lower().replace(" ", "").replace("**", "^")

    @staticmethod
    def _read_variable(variable):
        return np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan)

    def load(self):
        if self.data is not None:
            return self.data

        if not self.instant_file.exists():
            raise FileNotFoundError(f"Instant NetCDF file not found: {self.instant_file}")
        if not self.accum_file.exists():
            raise FileNotFoundError(f"Accumulated NetCDF file not found: {self.accum_file}")

        with Dataset(self.instant_file) as instant, Dataset(self.accum_file) as accum:
            for ds, label in ((instant, "Instant"), (accum, "Accumulated")):
                if "valid_time" not in ds.variables:
                    raise ValueError(f"{label} NetCDF tidak memiliki valid_time.")

            required_instant = ("t2m", "sp", "zust")
            required_accum = ("tp", "sshf")

            for name in required_instant:
                if name not in instant.variables:
                    raise ValueError(f"Variable {name} tidak ditemukan pada Surface Instant NetCDF.")

            for name in required_accum:
                if name not in accum.variables:
                    raise ValueError(f"Variable {name} tidak ditemukan pada Surface Accumulated NetCDF.")

            zust_units = self._normalize_units(
                getattr(instant.variables["zust"], "units", "")
            )
            if zust_units not in ("ms^-1", "ms-1", "m/s"):
                raise ValueError(f"Unsupported zust units: {getattr(instant.variables['zust'], 'units', '')}")

            instant_times = self._times(instant.variables["valid_time"])
            accum_times = self._times(accum.variables["valid_time"])

            if len(instant_times) != len(accum_times):
                raise ValueError("Jumlah waktu Surface Instant dan Accumulated berbeda.")

            for i, (t1, t2) in enumerate(zip(instant_times, accum_times)):
                if abs(self._seconds_between(t1,t2))>0.001:
                    raise ValueError(f"Valid time Instant dan Accumulated berbeda pada indeks {i}.")

            reference_times = self._times(accum.variables["time"]) if "time" in accum.variables else []
            reference_time = reference_times[0] if reference_times else None

            tp_variable = accum.variables["tp"]
            rh_variable = instant.variables.get("rh2m")

            self.data = {
                "times": accum_times,
                "reference_time": reference_time,
                "latitude": np.asarray(instant.variables["latitude"][:], dtype=float),
                "longitude": np.asarray(instant.variables["longitude"][:], dtype=float),
                "friction_velocity": self._read_variable(instant.variables["zust"]),
                "temperature": self._read_variable(instant.variables["t2m"]),
                "surface_pressure": self._read_variable(instant.variables["sp"]),
                "relative_humidity": self._read_variable(rh_variable) if rh_variable is not None else None,
                "relative_humidity_units": getattr(rh_variable, "units", "%") if rh_variable is not None else None,
                "precipitation": self._read_variable(tp_variable),
                "heat_flux": self._read_variable(accum.variables["sshf"]),
                "precipitation_units": getattr(tp_variable, "units", "m"),
                "precipitation_availability": getattr(tp_variable, "availability", "unknown"),
                "heat_flux_units": getattr(accum.variables["sshf"], "units", ""),
                "tp_start_step": self._read_variable(accum.variables["tp_start_step"]),
                "tp_end_step": self._read_variable(accum.variables["tp_end_step"]),
            }

        self._validate()
        return self.data

    def precipitation_index_at_elapsed(self, elapsed_seconds):
        data = self.load()
        elapsed = float(elapsed_seconds)

        times = data["times"]
        if not times:
            raise ValueError("Dataset Surface tidak memiliki waktu valid.")

        offsets = np.array([
            self._seconds_between(times[0], t)
            for t in times
        ], dtype=float)

        if elapsed < 0 or elapsed > offsets[-1]:
            raise ValueError("Waktu di luar cakupan Surface GFS.")

        if elapsed == offsets[-1]:
            return len(offsets) - 1

        return min(
            int(np.searchsorted(offsets, elapsed, side="right")),
            len(offsets) - 1
        )
    def _validate(self):
        data = self.data

        if not data["times"]:
            raise ValueError("Dataset Surface tidak memiliki waktu valid.")
        if not len(data["latitude"]) or not len(data["longitude"]):
            raise ValueError("Koordinat latitude/longitude kosong.")
        if not np.all(np.isfinite(data["latitude"])):
            raise ValueError("Latitude mengandung nilai tidak valid.")
        if not np.all(np.isfinite(data["longitude"])):
            raise ValueError("Longitude mengandung nilai tidak valid.")

        expected = (len(data["times"]), len(data["latitude"]), len(data["longitude"]))

        for name in ("friction_velocity", "temperature", "surface_pressure", "precipitation", "heat_flux"):
            if data[name].shape != expected:
                raise ValueError(
                    f"Unexpected shape for {name}. Expected {expected}, received {data[name].shape}"
                )

        if data["relative_humidity"] is not None:
            if data["relative_humidity"].shape != expected:
                raise ValueError(
                    f"Unexpected shape for relative_humidity. Expected {expected}, received {data['relative_humidity'].shape}"
                )

        tp_units = self._normalize_units(data["precipitation_units"])
        if tp_units not in ("m", "meter", "metre"):
            raise ValueError(f"Unsupported precipitation unit: {data['precipitation_units']}")

        heat_units = self._normalize_units(data["heat_flux_units"])
        if heat_units not in ("wm^-2", "wm-2", "w/m^2", "jm^-2", "jm-2", "j/m^2"):
            raise ValueError(f"Unsupported heat flux unit: {data['heat_flux_units']}")

        if not np.any(np.isfinite(data["friction_velocity"])):
            raise ValueError("Data friction velocity tidak tersedia.")
        if not np.any(np.isfinite(data["precipitation"])):
            raise ValueError("Data precipitation tidak tersedia.")

    def interval_seconds(self, time_index=0):
        data = self.load()
        times = data["times"]
        if not times:
            return 0.0

        index = max(0, min(int(time_index), len(times) - 1))
        if index > 0:
            return self._seconds_between(times[index - 1], times[index])
        if data["reference_time"] is not None:
            return self._seconds_between(data["reference_time"], times[index])
        return 0.0

    @staticmethod
    def precipitation_m_to_mm(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return max(value, 0.0) * 1000.0

    def precipitation_area(self, time_index, lat_index, lon_index, radius=2):
        data = self.load()
        tp = data["precipitation"][time_index]

        lat_start = max(0, lat_index - radius)
        lat_end = min(len(data["latitude"]), lat_index + radius + 1)
        lon_start = max(0, lon_index - radius)
        lon_end = min(len(data["longitude"]), lon_index + radius + 1)

        area = tp[lat_start:lat_end, lon_start:lon_end]
        area_mm = np.where(np.isfinite(area), np.maximum(area, 0.0) * 1000.0, np.nan)
        valid = area_mm[np.isfinite(area_mm)]

        grid = [
            [round(float(value), 6) if np.isfinite(value) else None for value in row]
            for row in area_mm
        ]

        return {
            "precipitation_area_available": bool(valid.size),
            "precipitation_area_grid": grid,
            "precipitation_area_grid_count": int(valid.size),
            "precipitation_area_max_mm": float(np.max(valid)) if valid.size else None,
            "precipitation_area_mean_mm": float(np.mean(valid)) if valid.size else None,
            "precipitation_area_min_mm": float(np.min(valid)) if valid.size else None,
        }

    def precipitation_rate(self,time_index,lat_index,lon_index,interval_seconds):
        data=self.load()
        tp=float(data["precipitation"][time_index,lat_index,lon_index])
        start=float(data["tp_start_step"][time_index])
        end=float(data["tp_end_step"][time_index])

        if not all(np.isfinite(v) for v in (tp,start,end)) or end<=start:
            return None

        duration=end-start

        if time_index>0:
            previous_start=float(data["tp_start_step"][time_index-1])
            previous_end=float(data["tp_end_step"][time_index-1])
            previous_tp=float(data["precipitation"][time_index-1,lat_index,lon_index])

            if (
                np.isfinite(previous_tp)
                and np.isfinite(previous_start)
                and np.isfinite(previous_end)
                and abs(previous_start-start)<1e-6
                and start<previous_end<end
            ):
                tp=max(tp-previous_tp,0.0)
                duration=end-previous_end

        return max(tp,0.0)*1000.0/duration

    def heat_flux_wm2(self, time_index, lat_index, lon_index, interval_seconds):
        data = self.load()
        value = float(data["heat_flux"][time_index, lat_index, lon_index])
        if not np.isfinite(value):
            return None

        units = self._normalize_units(data["heat_flux_units"])
        if units in ("wm^-2", "wm-2", "w/m^2"):
            return value

        if units in ("jm^-2", "jm-2", "j/m^2"):
            if interval_seconds <= 0:
                return None
            if time_index > 0:
                previous = float(data["heat_flux"][time_index - 1, lat_index, lon_index])
                if not np.isfinite(previous):
                    return None
                value -= previous
            return value / interval_seconds

        raise ValueError(f"Unsupported heat flux units: {data['heat_flux_units']}")

    def time_index_at_elapsed(self, elapsed_seconds):
        data = self.load()
        times = data["times"]
        if not times:
            raise ValueError("Dataset Surface tidak memiliki waktu valid.")

        elapsed = np.array([
            self._seconds_between(times[0], t) for t in times
        ], dtype=float)
        target = float(elapsed_seconds)

        if not np.isfinite(target) or target < 0 or target > elapsed[-1]:
            raise ValueError("Waktu simulasi di luar cakupan Surface GFS.")

        return int(np.searchsorted(elapsed, target, side="right") - 1)

    def surface_profile(self, latitude, longitude, time_index=0,
                    precipitation_index=None):
        data = self.load()
        if not data["times"]:
            raise ValueError("Dataset Surface tidak memiliki waktu valid.")

        lat_index = int(np.abs(data["latitude"] - float(latitude)).argmin())
        lon_index = int(np.abs(data["longitude"] - float(longitude)).argmin())
        time_index = max(0, min(int(time_index), len(data["times"]) - 1))
        rain_index = time_index if precipitation_index is None else int(precipitation_index)

        if not 0 <= rain_index < len(data["times"]):
            raise ValueError("Indeks presipitasi GFS tidak valid.")
        interval = self.interval_seconds(time_index)

        friction_velocity = float(data["friction_velocity"][time_index, lat_index, lon_index])
        if not np.isfinite(friction_velocity) or friction_velocity <= 0:
            raise ValueError(
                f"Invalid friction velocity at {data['latitude'][lat_index]}, {data['longitude'][lon_index]}."
            )

        temperature = float(data["temperature"][time_index, lat_index, lon_index])
        pressure = float(data["surface_pressure"][time_index, lat_index, lon_index])
        if not np.isfinite(temperature) or not np.isfinite(pressure):
            raise ValueError("Temperature atau surface pressure tidak valid.")

        rh = None
        if data["relative_humidity"] is not None:
            value = float(data["relative_humidity"][time_index, lat_index, lon_index])
            if np.isfinite(value):
                rh = float(np.clip(value, 0.0, 100.0))

        tp = float(data["precipitation"][rain_index, lat_index, lon_index])
        precipitation_available = bool(np.isfinite(tp))
        precipitation_mm = self.precipitation_m_to_mm(tp)

        precipitation_rate = self.precipitation_rate(
            rain_index,
            lat_index,
            lon_index,
            self.interval_seconds(rain_index)
        )

        area_stats = self.precipitation_area(
            rain_index, lat_index, lon_index, radius=2
        )

        heat_flux = float(data["heat_flux"][time_index, lat_index, lon_index])
        heat_flux_wm2 = self.heat_flux_wm2(
            time_index, lat_index, lon_index, interval
        )

        return {
            "latitude": float(data["latitude"][lat_index]),
            "longitude": float(data["longitude"][lon_index]),
            "time_index": int(time_index),
            "time": str(data["times"][time_index]),
            "interval_seconds": float(interval),
            "interval_hours": float(interval / 3600.0) if interval > 0 else None,
            "friction_velocity": friction_velocity,
            "temperature": temperature,
            "surface_pressure": pressure,
            "relative_humidity": rh,
            "relative_humidity_units": data["relative_humidity_units"],
            "precipitation_available": precipitation_available,
            "precipitation": tp if precipitation_available else None,
            "precipitation_m": tp if precipitation_available else None,
            "precipitation_mm": precipitation_mm,
            "precipitation_rate_mm_hour": precipitation_rate,
            "precipitation_units": data["precipitation_units"],
            **area_stats,
            "heat_flux_accumulated": heat_flux,
            "heat_flux_wm2": heat_flux_wm2,
            "heat_flux_units": data["heat_flux_units"],
        }

    def close(self):
        self.data = None
