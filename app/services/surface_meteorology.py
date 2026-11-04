
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
                    only_use_python_datetimes=True
                )
                return list(np.atleast_1d(result))
            except Exception:
                pass
        return list(np.atleast_1d(values))

    @staticmethod
    def _seconds_between(start, end):
        try:
            return max(0.0, float((end - start).total_seconds()))
        except Exception:
            try:
                return max(0.0, float((end - start) / np.timedelta64(1, "s")))
            except Exception:
                return 0.0

    @staticmethod
    def _normalize_units(units):
        return str(units or "").strip().lower().replace(" ", "").replace("**", "^")

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

            instant_times = self._times(instant.variables["valid_time"])
            accum_times = self._times(accum.variables["valid_time"])

            if len(instant_times) != len(accum_times):
                raise ValueError("Jumlah waktu Surface Instant dan Accumulated berbeda.")

            for i, (t1, t2) in enumerate(zip(instant_times, accum_times)):
                if self._seconds_between(t1, t2) != 0:
                    raise ValueError(f"Valid time Instant dan Accumulated berbeda pada indeks {i}.")

            reference_times = self._times(accum.variables["time"]) if "time" in accum.variables else []
            reference_time = reference_times[0] if reference_times else None

            self.data = {
                "times": accum_times,
                "reference_time": reference_time,
                "latitude": np.asarray(instant.variables["latitude"][:], dtype=float),
                "longitude": np.asarray(instant.variables["longitude"][:], dtype=float),
                "friction_velocity": np.asarray(instant.variables["zust"][:], dtype=float),
                "temperature": np.asarray(instant.variables["t2m"][:], dtype=float),
                "surface_pressure": np.asarray(instant.variables["sp"][:], dtype=float),
                "precipitation": np.asarray(accum.variables["tp"][:], dtype=float),
                "heat_flux": np.asarray(accum.variables["sshf"][:], dtype=float),
                "precipitation_units": getattr(accum.variables["tp"], "units", "m"),
                "heat_flux_units": getattr(accum.variables["sshf"], "units", "")
            }

        self._validate()
        return self.data

    def _validate(self):
        data = self.data
        expected = (
            len(data["times"]),
            len(data["latitude"]),
            len(data["longitude"])
        )

        for name in (
            "friction_velocity",
            "temperature",
            "surface_pressure",
            "precipitation",
            "heat_flux"
        ):
            if data[name].shape != expected:
                raise ValueError(
                    f"Unexpected shape for {name}. "
                    f"Expected {expected}, received {data[name].shape}"
                )

        tp_units = self._normalize_units(data["precipitation_units"])
        if tp_units not in ("m", "meter", "metre"):
            raise ValueError(f"Unsupported precipitation unit: {data['precipitation_units']}")

        heat_units = self._normalize_units(data["heat_flux_units"])
        if heat_units not in ("wm^-2", "wm-2", "w/m^2", "jm^-2", "jm-2", "j/m^2"):
            raise ValueError(f"Unsupported heat flux unit: {data['heat_flux_units']}")

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
        return max(float(value), 0.0) * 1000.0

    def precipitation_rate(self, time_index, lat_index, lon_index, interval_seconds):
        data = self.load()
        current = max(float(data["precipitation"][time_index, lat_index, lon_index]), 0.0)

        # tp GFS merupakan akumulasi sejak forecast reference time.
        # Untuk timestep berikutnya, hitung selisih akumulasi.
        if time_index > 0:
            previous = max(float(data["precipitation"][time_index - 1, lat_index, lon_index]), 0.0)
            current = max(current - previous, 0.0)

        interval_hours = float(interval_seconds) / 3600.0
        if interval_hours <= 0:
            return 0.0

        return self.precipitation_m_to_mm(current) / interval_hours

    def heat_flux_wm2(self, time_index, lat_index, lon_index, interval_seconds):
        data = self.load()
        value = float(data["heat_flux"][time_index, lat_index, lon_index])
        units = self._normalize_units(data["heat_flux_units"])

        # GFS avg_ishf sudah merupakan time-mean flux dalam W/m2.
        if units in ("wm^-2", "wm-2", "w/m^2"):
            return value

        # Jika sumber berupa akumulasi energi J/m2, ubah menjadi W/m2.
        if interval_seconds <= 0:
            return 0.0

        if time_index > 0:
            previous = float(data["heat_flux"][time_index - 1, lat_index, lon_index])
            value -= previous

        return value / float(interval_seconds)

    def time_index_at_elapsed(self, elapsed_seconds):
        data = self.load()
        if not data["times"]:
            return 0

        first = data["times"][0]
        elapsed = [
            self._seconds_between(first, current)
            for current in data["times"]
        ]

        return int(
            np.abs(np.asarray(elapsed, dtype=float) - float(elapsed_seconds)).argmin()
        )

    def surface_profile(self, latitude, longitude, time_index=0):
        data = self.load()

        lat_index = int(np.abs(data["latitude"] - float(latitude)).argmin())
        lon_index = int(np.abs(data["longitude"] - float(longitude)).argmin())
        time_index = max(0, min(int(time_index), len(data["times"]) - 1))

        interval = self.interval_seconds(time_index)
        tp = max(float(data["precipitation"][time_index, lat_index, lon_index]), 0.0)
        precipitation_mm = self.precipitation_m_to_mm(tp)
        precipitation_rate = self.precipitation_rate(
            time_index, lat_index, lon_index, interval
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
            "friction_velocity": float(data["friction_velocity"][time_index, lat_index, lon_index]),
            "temperature": float(data["temperature"][time_index, lat_index, lon_index]),
            "surface_pressure": float(data["surface_pressure"][time_index, lat_index, lon_index]),
            "precipitation": tp,
            "precipitation_m": tp,
            "precipitation_mm": precipitation_mm,
            "precipitation_rate_mm_hour": precipitation_rate,
            "precipitation_units": data["precipitation_units"],
            "heat_flux_accumulated": heat_flux,
            "heat_flux_wm2": heat_flux_wm2,
            "heat_flux_units": data["heat_flux_units"]
        }

    def close(self):
        self.data = None
