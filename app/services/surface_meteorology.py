
from pathlib import Path
from functools import lru_cache

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

        if not hasattr(variable, "units"):
            raise ValueError("Metadata satuan waktu tidak tersedia.")

        result = num2date(
            values,
            units=variable.units,
            calendar=getattr(variable, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )

        return list(np.atleast_1d(result))

    @staticmethod
    def _seconds_between(start, end):
        try:
            return float((end - start).total_seconds())
        except (AttributeError, TypeError):
            return float(
                (end - start) / np.timedelta64(1, "s")
            )

    @staticmethod
    def _normalize_units(units):
        return (
            str(units or "")
            .strip()
            .lower()
            .replace(" ", "")
            .replace("**", "^")
        )

    @staticmethod
    def _read_variable(variable):
        return np.ma.filled(
            np.ma.asarray(variable[:], dtype=float),
            np.nan
        )

    def load(self):
        if self.data is not None:
            return self.data

        if not self.instant_file.exists():
            raise FileNotFoundError(
                f"Instant NetCDF file not found: {self.instant_file}"
            )

        if not self.accum_file.exists():
            raise FileNotFoundError(
                f"Accumulated NetCDF file not found: {self.accum_file}"
            )

        with Dataset(self.instant_file) as instant, \
                Dataset(self.accum_file) as accum:

            for ds, label in (
                (instant, "Instant"),
                (accum, "Accumulated"),
            ):
                for coordinate in (
                    "valid_time",
                    "latitude",
                    "longitude",
                ):
                    if coordinate not in ds.variables:
                        raise ValueError(
                            f"{label} NetCDF tidak memiliki "
                            f"{coordinate}."
                        )

            required_instant = (
                "t2m",
                "sp",
                "zust",
            )

            required_accum = (
                "tp",
                "sshf",
                "tp_start_step",
                "tp_end_step",
                "sshf_start_step",
                "sshf_end_step",
            )

            for name in required_instant:
                if name not in instant.variables:
                    raise ValueError(
                        f"Variable {name} tidak ditemukan "
                        "pada Surface Instant NetCDF."
                    )

            for name in required_accum:
                if name not in accum.variables:
                    raise ValueError(
                        f"Variable {name} tidak ditemukan "
                        "pada Surface Accumulated NetCDF."
                    )

            zust_units = self._normalize_units(
                getattr(
                    instant.variables["zust"],
                    "units",
                    ""
                )
            )

            if zust_units not in (
                "ms^-1",
                "ms-1",
                "m/s",
            ):
                raise ValueError(
                    f"Unsupported zust units: {zust_units}"
                )

            instant_times = self._times(
                instant.variables["valid_time"]
            )

            accum_times = self._times(
                accum.variables["valid_time"]
            )

            for times, label in (
                (instant_times, "Instant"),
                (accum_times, "Accumulated"),
            ):
                if not times:
                    raise ValueError(
                        f"Surface {label} tidak memiliki waktu valid."
                    )

                for previous, current in zip(
                    times,
                    times[1:]
                ):
                    if self._seconds_between(
                        previous, current
                    ) <= 0:
                        raise ValueError(
                            f"Waktu Surface {label} tidak berurutan."
                        )

            for coordinate in (
                "latitude",
                "longitude",
            ):
                instant_values = np.asarray(
                    instant.variables[coordinate][:],
                    dtype=float
                )

                accum_values = np.asarray(
                    accum.variables[coordinate][:],
                    dtype=float
                )

                if (
                    instant_values.shape != accum_values.shape
                    or not np.allclose(
                        instant_values,
                        accum_values,
                        rtol=0,
                        atol=1e-6,
                    )
                ):
                    raise ValueError(
                        f"Grid {coordinate} Instant "
                        "dan Accumulated berbeda."
                    )

            reference_times = (
                self._times(accum.variables["time"])
                if "time" in accum.variables
                else []
            )

            reference_time = (
                reference_times[0]
                if reference_times
                else None
            )

            tp_variable = accum.variables["tp"]
            rh_variable = instant.variables.get("rh2m")
            heat_variable = accum.variables["sshf"]

            expected_dimensions = (
                "valid_time",
                "latitude",
                "longitude",
            )

            for ds, names in (
                (instant, required_instant),
                (accum, ("tp", "sshf")),
            ):
                for name in names:
                    if ds.variables[name].dimensions != expected_dimensions:
                        raise ValueError(
                            f"Dimensi {name} tidak sesuai: "
                            f"{ds.variables[name].dimensions}"
                        )

            if rh_variable is not None:
                if rh_variable.dimensions != expected_dimensions:
                    raise ValueError(
                        "Dimensi relative humidity tidak sesuai."
                    )

            for name in (
                "tp_start_step",
                "tp_end_step",
                "sshf_start_step",
                "sshf_end_step",
            ):
                variable = accum.variables[name]

                if variable.dimensions != ("valid_time",):
                    raise ValueError(
                        f"Dimensi metadata {name} tidak sesuai."
                    )

                units = self._normalize_units(
                    getattr(variable, "units", "")
                )

                if units not in ("h", "hr", "hour", "hours"):
                    raise ValueError(
                        f"Satuan interval {name} tidak didukung: "
                        f"{units}"
                    )

            self.data = {
                "times": instant_times,
                "instant_times": instant_times,
                "accum_times": accum_times,
                "reference_time": reference_time,

                "latitude": np.asarray(
                    instant.variables["latitude"][:],
                    dtype=float
                ),

                "longitude": np.asarray(
                    instant.variables["longitude"][:],
                    dtype=float
                ),

                "friction_velocity": self._read_variable(
                    instant.variables["zust"]
                ),

                "temperature": self._read_variable(
                    instant.variables["t2m"]
                ),

                "surface_pressure": self._read_variable(
                    instant.variables["sp"]
                ),

                "relative_humidity": (
                    self._read_variable(rh_variable)
                    if rh_variable is not None
                    else None
                ),

                "relative_humidity_units": (
                    getattr(rh_variable, "units", "%")
                    if rh_variable is not None
                    else None
                ),

                "precipitation": self._read_variable(
                    tp_variable
                ),

                "heat_flux": self._read_variable(
                    heat_variable
                ),

                "precipitation_units": getattr(
                    tp_variable,
                    "units",
                    "m"
                ),

                "precipitation_availability": getattr(
                    tp_variable,
                    "availability",
                    "unknown"
                ),

                "heat_flux_units": getattr(
                    heat_variable,
                    "units",
                    ""
                ),

                "tp_start_step": self._read_variable(
                    accum.variables["tp_start_step"]
                ),

                "tp_end_step": self._read_variable(
                    accum.variables["tp_end_step"]
                ),

                "sshf_start_step": self._read_variable(
                    accum.variables["sshf_start_step"]
                ),

                "sshf_end_step": self._read_variable(
                    accum.variables["sshf_end_step"]
                ),
            }

        try:
            self._validate()
        except Exception:
            self.data = None
            raise

        return self.data

    def _validate(self):
        data = self.data

        if not data["instant_times"]:
            raise ValueError(
                "Dataset Surface Instant tidak memiliki waktu valid."
            )

        if not data["accum_times"]:
            raise ValueError(
                "Dataset Surface Accumulated tidak memiliki waktu valid."
            )

        for coordinate in (
            "latitude",
            "longitude",
        ):
            values = data[coordinate]

            if values.ndim != 1 or not values.size:
                raise ValueError(
                    f"Koordinat {coordinate} tidak valid."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{coordinate} mengandung nilai tidak valid."
                )

            if len(np.unique(values)) != len(values):
                raise ValueError(
                    f"Koordinat {coordinate} mengandung duplikasi."
                )

        spatial_shape = (
            len(data["latitude"]),
            len(data["longitude"]),
        )

        expected_instant = (
            len(data["instant_times"]),
            *spatial_shape,
        )

        expected_accum = (
            len(data["accum_times"]),
            *spatial_shape,
        )

        for name in (
            "friction_velocity",
            "temperature",
            "surface_pressure",
        ):
            if data[name].shape != expected_instant:
                raise ValueError(
                    f"Unexpected shape for {name}. "
                    f"Expected {expected_instant}, "
                    f"received {data[name].shape}"
                )

        for name in (
            "precipitation",
            "heat_flux",
        ):
            if data[name].shape != expected_accum:
                raise ValueError(
                    f"Unexpected shape for {name}. "
                    f"Expected {expected_accum}, "
                    f"received {data[name].shape}"
                )

        if data["relative_humidity"] is not None:
            if data["relative_humidity"].shape != expected_instant:
                raise ValueError(
                    "Unexpected shape for relative_humidity."
                )

        for name in (
            "tp_start_step",
            "tp_end_step",
            "sshf_start_step",
            "sshf_end_step",
        ):
            if data[name].shape != (
                len(data["accum_times"]),
            ):
                raise ValueError(
                    f"Unexpected shape for {name}: "
                    f"{data[name].shape}"
                )

        tp_units = self._normalize_units(
            data["precipitation_units"]
        )

        if tp_units not in (
            "m",
            "meter",
            "metre",
        ):
            raise ValueError(
                f"Unsupported precipitation unit: "
                f"{data['precipitation_units']}"
            )

        heat_units = self._normalize_units(
            data["heat_flux_units"]
        )

        if heat_units not in (
            "wm^-2",
            "wm-2",
            "w/m^2",
            "jm^-2",
            "jm-2",
            "j/m^2",
        ):
            raise ValueError(
                f"Unsupported heat flux unit: "
                f"{data['heat_flux_units']}"
            )

        for name in (
            "friction_velocity",
            "temperature",
            "surface_pressure",
            "precipitation",
            "heat_flux",
        ):
            if not np.any(np.isfinite(data[name])):
                raise ValueError(
                    f"Data {name} tidak tersedia."
                )

        self._validate_interval_coverage("tp")
        self._validate_interval_coverage("sshf")

    def _validate_interval_coverage(self, variable):
        data = self.data

        starts = data[f"{variable}_start_step"]
        ends = data[f"{variable}_end_step"]

        origin = data["instant_times"][0]

        coverage = self._seconds_between(
            origin,
            data["instant_times"][-1]
        )

        intervals = []
        reference_offsets = []

        for index, valid_time in enumerate(
            data["accum_times"]
        ):
            start = float(starts[index])
            end = float(ends[index])

            if not all(
                np.isfinite(v)
                for v in (start, end)
            ):
                raise ValueError(
                    f"Metadata {variable} tidak valid "
                    f"pada indeks {index}."
                )

            if start < 0 or end <= start:
                raise ValueError(
                    f"Interval {variable} tidak valid: "
                    f"{start}-{end}."
                )

            interval_end = self._seconds_between(
                origin,
                valid_time
            )

            duration = (end - start) * 3600.0

            interval_start = interval_end - duration

            reference_offsets.append(
                interval_end - end * 3600.0
            )

            intervals.append(
                (interval_start, interval_end)
            )

        if not intervals:
            raise ValueError(
                f"Interval {variable} kosong."
            )

        tolerance = 1e-3

        if not np.allclose(
            reference_offsets,
            reference_offsets[0],
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(
                f"Forecast reference time {variable} tidak konsisten."
            )

        if intervals[0][0] > tolerance:
            raise ValueError(
                f"Cakupan awal {variable} tidak mencukupi."
            )

        for previous, current in zip(
            intervals,
            intervals[1:]
        ):
            difference = current[0] - previous[1]

            if difference > tolerance:
                raise ValueError(
                    f"Terdapat celah waktu "
                    f"pada interval {variable}."
                )

            if difference < -tolerance:
                raise ValueError(
                    f"Interval {variable} saling tumpang tindih. "
                    "Normalisasi accumulated diperlukan."
                )

        if intervals[-1][1] < coverage - tolerance:
            raise ValueError(
                f"Cakupan akhir {variable} tidak mencukupi."
            )

        data[f"{variable}_intervals"] = np.asarray(
            intervals,
            dtype=float
        )

    def _validate_elapsed(self, elapsed_seconds):
        data = self.load()

        elapsed = float(elapsed_seconds)

        coverage = self._seconds_between(
            data["instant_times"][0],
            data["instant_times"][-1]
        )

        if not np.isfinite(elapsed):
            raise ValueError(
                "Waktu simulasi tidak valid."
            )

        if elapsed < 0 or elapsed > coverage:
            raise ValueError(
                f"Waktu simulasi {elapsed} detik "
                "di luar cakupan Surface Instant."
            )

        return elapsed

    def _accum_index_at_elapsed(
        self,
        variable,
        elapsed_seconds
    ):
        data = self.load()

        elapsed = self._validate_elapsed(
            elapsed_seconds
        )

        intervals = data[f"{variable}_intervals"]

        index = int(
            np.searchsorted(
                intervals[:, 1],
                elapsed,
                side="right"
            )
        )

        if index == len(intervals):
            if abs(
                elapsed - intervals[-1, 1]
            ) <= 1e-6:
                return len(intervals) - 1

            raise ValueError(
                f"Waktu di luar cakupan {variable}."
            )

        start, end = intervals[index]

        if start <= elapsed < end:
            return index

        raise ValueError(
            f"Interval {variable} tidak tersedia "
            f"pada elapsed={elapsed} detik."
        )

    def precipitation_index_at_elapsed(
        self,
        elapsed_seconds
    ):
        return self._accum_index_at_elapsed(
            "tp",
            elapsed_seconds
        )

    def heat_flux_index_at_elapsed(
        self,
        elapsed_seconds
    ):
        return self._accum_index_at_elapsed(
            "sshf",
            elapsed_seconds
        )

    def _interval_seconds(
        self,
        variable,
        time_index
    ):
        data = self.load()

        index = int(time_index)

        intervals = data[f"{variable}_intervals"]

        if not 0 <= index < len(intervals):
            raise ValueError(
                f"Indeks interval {variable} tidak valid."
            )

        start, end = intervals[index]

        duration = float(end - start)

        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Durasi interval {variable} tidak valid."
            )

        return duration

    def interval_seconds(self, time_index=0):
        return self._interval_seconds(
            "tp",
            time_index
        )

    def heat_flux_interval_seconds(self, time_index):
        return self._interval_seconds(
            "sshf",
            time_index
        )

    @staticmethod
    def precipitation_m_to_mm(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(value):
            return None

        return max(value, 0.0) * 1000.0

    def precipitation_area(
        self,
        time_index,
        lat_index,
        lon_index,
        radius=2
    ):
        data = self.load()

        tp = data["precipitation"][time_index]

        lat_start = max(
            0,
            lat_index - radius
        )

        lat_end = min(
            len(data["latitude"]),
            lat_index + radius + 1
        )

        lon_start = max(
            0,
            lon_index - radius
        )

        lon_end = min(
            len(data["longitude"]),
            lon_index + radius + 1
        )

        area = tp[
            lat_start:lat_end,
            lon_start:lon_end
        ]

        area_mm = np.where(
            np.isfinite(area),
            np.maximum(area, 0.0) * 1000.0,
            np.nan
        )

        valid = area_mm[np.isfinite(area_mm)]

        grid = [
            [
                round(float(value), 6)
                if np.isfinite(value)
                else None
                for value in row
            ]
            for row in area_mm
        ]

        return {
            "precipitation_area_available": bool(
                valid.size
            ),

            "precipitation_area_grid": grid,

            "precipitation_area_grid_count": int(
                valid.size
            ),

            "precipitation_area_max_mm": (
                float(np.max(valid))
                if valid.size
                else None
            ),

            "precipitation_area_mean_mm": (
                float(np.mean(valid))
                if valid.size
                else None
            ),

            "precipitation_area_min_mm": (
                float(np.min(valid))
                if valid.size
                else None
            ),
        }

    def precipitation_rate(
        self,
        time_index,
        lat_index,
        lon_index,
        interval_seconds
    ):
        data = self.load()

        tp = float(
            data["precipitation"][
                time_index,
                lat_index,
                lon_index
            ]
        )

        if not np.isfinite(tp):
            return None

        duration = self.interval_seconds(
            time_index
        )

        if duration <= 0:
            return None

        duration_hours = duration / 3600.0

        return (
            max(tp, 0.0)
            * 1000.0
            / duration_hours
        )

    def heat_flux_wm2(
        self,
        time_index,
        lat_index,
        lon_index,
        interval_seconds
    ):
        data = self.load()

        value = float(
            data["heat_flux"][
                time_index,
                lat_index,
                lon_index
            ]
        )

        if not np.isfinite(value):
            return None

        units = self._normalize_units(
            data["heat_flux_units"]
        )

        if units in (
            "wm^-2",
            "wm-2",
            "w/m^2",
        ):
            return value

        if units in (
            "jm^-2",
            "jm-2",
            "j/m^2",
        ):
            duration = self.heat_flux_interval_seconds(
                time_index
            )

            if duration <= 0:
                return None

            return value / duration

        raise ValueError(
            f"Unsupported heat flux units: "
            f"{data['heat_flux_units']}"
        )

    def time_index_at_elapsed(
        self,
        elapsed_seconds
    ):
        data = self.load()

        target = self._validate_elapsed(
            elapsed_seconds
        )

        times = data["instant_times"]

        elapsed = np.asarray([
            self._seconds_between(
                times[0],
                time
            )
            for time in times
        ], dtype=float)

        return int(
            np.searchsorted(
                elapsed,
                target,
                side="right"
            ) - 1
        )

    def surface_profile(
        self,
        latitude,
        longitude,
        time_index=0,
        precipitation_index=None,
        elapsed_seconds=None,
    ):
        data = self.load()

        instant_times = data["instant_times"]
        accum_times = data["accum_times"]

        latitude = float(latitude)
        longitude = float(longitude)

        if not all(
            np.isfinite(v)
            for v in (latitude, longitude)
        ):
            raise ValueError(
                "Koordinat Surface tidak valid."
            )

        if not (
            np.min(data["latitude"])
            <= latitude
            <= np.max(data["latitude"])
        ):
            raise ValueError(
                "Latitude di luar cakupan Surface."
            )

        if not (
            np.min(data["longitude"])
            <= longitude
            <= np.max(data["longitude"])
        ):
            raise ValueError(
                "Longitude di luar cakupan Surface."
            )

        lat_index = int(
            np.abs(
                data["latitude"] - latitude
            ).argmin()
        )

        lon_index = int(
            np.abs(
                data["longitude"] - longitude
            ).argmin()
        )

        if elapsed_seconds is None:
            time_index = int(time_index)

            if not 0 <= time_index < len(instant_times):
                raise ValueError(
                    "Indeks Surface Instant tidak valid."
                )

            elapsed = self._seconds_between(
                instant_times[0],
                instant_times[time_index]
            )

        else:
            elapsed = self._validate_elapsed(
                elapsed_seconds
            )

            time_index = self.time_index_at_elapsed(
                elapsed
            )

        expected_rain_index = (
            self.precipitation_index_at_elapsed(
                elapsed
            )
        )

        if precipitation_index is None:
            rain_index = expected_rain_index

        else:
            rain_index = int(precipitation_index)

            if not 0 <= rain_index < len(accum_times):
                raise ValueError(
                    "Indeks presipitasi tidak valid."
                )

            if rain_index != expected_rain_index:
                raise ValueError(
                    "Indeks presipitasi tidak sesuai "
                    "dengan waktu simulasi."
                )

        heat_index = self.heat_flux_index_at_elapsed(
            elapsed
        )

        result = self._surface_profile_cached(
            lat_index,
            lon_index,
            time_index,
            rain_index,
            heat_index
        )

        return {
            **result,

            "precipitation_area_grid": [
                row.copy()
                for row in result[
                    "precipitation_area_grid"
                ]
            ],
        }

    @lru_cache(maxsize=8192)
    def _surface_profile_cached(
        self,
        lat_index,
        lon_index,
        time_index,
        rain_index,
        heat_index
    ):
        data = self.load()

        interval = self.interval_seconds(
            rain_index
        )

        heat_interval = self.heat_flux_interval_seconds(
            heat_index
        )

        friction_velocity = float(
            data["friction_velocity"][
                time_index,
                lat_index,
                lon_index
            ]
        )

        if (
            not np.isfinite(friction_velocity)
            or friction_velocity <= 0
        ):
            raise ValueError(
                "Invalid friction velocity at "
                f"{data['latitude'][lat_index]}, "
                f"{data['longitude'][lon_index]}."
            )

        temperature = float(
            data["temperature"][
                time_index,
                lat_index,
                lon_index
            ]
        )

        pressure = float(
            data["surface_pressure"][
                time_index,
                lat_index,
                lon_index
            ]
        )

        if (
            not np.isfinite(temperature)
            or not np.isfinite(pressure)
            or temperature <= 0
            or pressure <= 0
        ):
            raise ValueError(
                "Temperature atau surface pressure tidak valid."
            )

        rh = None

        if data["relative_humidity"] is not None:
            value = float(
                data["relative_humidity"][
                    time_index,
                    lat_index,
                    lon_index
                ]
            )

            if np.isfinite(value):
                rh = float(
                    np.clip(value, 0.0, 100.0)
                )

        tp = float(
            data["precipitation"][
                rain_index,
                lat_index,
                lon_index
            ]
        )

        precipitation_available = bool(
            np.isfinite(tp)
        )

        precipitation_mm = (
            self.precipitation_m_to_mm(tp)
        )

        precipitation_rate = self.precipitation_rate(
            rain_index,
            lat_index,
            lon_index,
            interval
        )

        area_stats = self.precipitation_area(
            rain_index,
            lat_index,
            lon_index,
            radius=2
        )

        heat_flux = float(
            data["heat_flux"][
                heat_index,
                lat_index,
                lon_index
            ]
        )

        heat_flux_wm2 = self.heat_flux_wm2(
            heat_index,
            lat_index,
            lon_index,
            heat_interval
        )

        return {
            "latitude": float(
                data["latitude"][lat_index]
            ),

            "longitude": float(
                data["longitude"][lon_index]
            ),

            "time_index": int(time_index),

            "time": str(
                data["instant_times"][time_index]
            ),

            "precipitation_index": int(
                rain_index
            ),

            "precipitation_time": str(
                data["accum_times"][rain_index]
            ),

            "heat_flux_index": int(
                heat_index
            ),

            "heat_flux_time": str(
                data["accum_times"][heat_index]
            ),

            "interval_seconds": float(
                interval
            ),

            "interval_hours": float(
                interval / 3600.0
            ),

            "heat_flux_interval_seconds": float(
                heat_interval
            ),

            "heat_flux_interval_hours": float(
                heat_interval / 3600.0
            ),

            "friction_velocity": friction_velocity,

            "temperature": temperature,

            "surface_pressure": pressure,

            "relative_humidity": rh,

            "relative_humidity_units": (
                data["relative_humidity_units"]
            ),

            "precipitation_available": (
                precipitation_available
            ),

            "precipitation": (
                tp
                if precipitation_available
                else None
            ),

            "precipitation_m": (
                tp
                if precipitation_available
                else None
            ),

            "precipitation_mm": precipitation_mm,

            "precipitation_rate_mm_hour": (
                precipitation_rate
            ),

            "precipitation_units": (
                data["precipitation_units"]
            ),

            **area_stats,

            "heat_flux_accumulated": (
                heat_flux
                if np.isfinite(heat_flux)
                else None
            ),

            "heat_flux_wm2": heat_flux_wm2,

            "heat_flux_units": data["heat_flux_units"],
        }

    def close(self):
        self._surface_profile_cached.cache_clear()
        self.data = None
