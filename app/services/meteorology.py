
from pathlib import Path
from datetime import datetime
import math
import numpy as np
from netCDF4 import Dataset, num2date


class MeteorologyService:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.dataset = None
        self.data = None
        self._time_info = None

    @staticmethod
    def _array(variable):
        values = np.ma.asarray(variable[:], dtype=float)
        return np.asarray(values.filled(np.nan), dtype=float)

    @staticmethod
    def _valid(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _geopotential_to_height(values, variable):
        if values is None:
            return None

        units = str(getattr(variable, "units", "")).lower()
        units = units.replace(" ", "").replace("**", "^")

        if units in (
            "m^2s^-2", "m2s-2", "m^2/s^2",
            "m2/s2", "j/kg"
        ):
            return values / 9.80665

        if units in (
            "m", "meter", "metre",
            "meters", "metres", "gpm"
        ):
            return values

        raise ValueError(
            f"Unsupported geopotential units: {units!r}"
        )

    def load(self):
        if self.data is not None:
            return self.data

        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"NetCDF file not found: {self.file_path}"
            )

        self.dataset = Dataset(self.file_path)

        try:
            variables = self.dataset.variables

            required = (
                "latitude", "longitude", "pressure_level",
                "valid_time", "u", "v"
            )

            missing = [
                name for name in required
                if name not in variables
            ]

            if missing:
                raise ValueError(
                    "Missing required NetCDF variables: "
                    + ", ".join(missing)
                )

            time_variable = variables["valid_time"]
            time_values = self._array(time_variable)

            if not hasattr(time_variable, "units"):
                raise ValueError(
                    "Meteorological time units are missing."
                )

            times = list(num2date(
                time_values,
                units=time_variable.units,
                calendar=getattr(
                    time_variable, "calendar", "standard"
                ),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True
            ))

            temperature_variable = variables.get(
                "temperature", variables.get("t")
            )

            geopotential_variable = variables.get(
                "geopotential", variables.get("z")
            )

            geopotential_values = (
                self._array(geopotential_variable)
                if geopotential_variable is not None
                else None
            )

            altitude_values = self._geopotential_to_height(
                geopotential_values,
                geopotential_variable
            )

            self.data = {
                "latitude": self._array(variables["latitude"]),
                "longitude": self._array(variables["longitude"]),
                "pressure_level": self._array(
                    variables["pressure_level"]
                ),
                "time": time_values,
                "times": times,
                "u": self._array(variables["u"]),
                "v": self._array(variables["v"]),
                "temperature": (
                    self._array(temperature_variable)
                    if temperature_variable is not None
                    else None
                ),
                "geopotential": geopotential_values,
                "altitude": altitude_values
            }

            self._validate()
            return self.data

        except Exception:
            self.close()
            raise

    def _validate(self):
        data = self.data
        u = data["u"]
        v = data["v"]

        if u.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Wind dimensions must be "
                "(time, pressure, latitude, longitude)."
            )

        if u.shape != v.shape:
            raise ValueError(
                "U and V wind dimensions do not match."
            )

        expected = (
            len(data["times"]),
            len(data["pressure_level"]),
            len(data["latitude"]),
            len(data["longitude"])
        )

        if u.shape != expected:
            raise ValueError(
                f"Invalid wind dimensions: {u.shape}, "
                f"expected {expected}."
            )

        if any(size == 0 for size in expected):
            raise ValueError(
                "Meteorological dataset contains empty dimensions."
            )

        if expected[1] < 2:
            raise ValueError(
                "At least two pressure levels are required."
            )

        if expected[2] < 2 or expected[3] < 2:
            raise ValueError(
                "At least two latitude and longitude "
                "grid points are required."
            )

        if data["altitude"] is None:
            raise ValueError(
                "Geopotential or altitude data is required."
            )

        for name in (
            "temperature", "geopotential", "altitude"
        ):
            values = data[name]

            if values is not None and values.shape != u.shape:
                raise ValueError(
                    f"{name} dimensions do not match wind data."
                )

        for name in ("latitude", "longitude"):
            axis = data[name]

            if not np.all(np.isfinite(axis)):
                raise ValueError(
                    f"Invalid {name} coordinates."
                )

            differences = np.diff(axis)

            if not (
                np.all(differences > 0)
                or np.all(differences < 0)
            ):
                raise ValueError(
                    f"{name} coordinates must be monotonic."
                )

        elapsed = []

        first = data["times"][0]

        for current in data["times"]:
            seconds = float(
                (current - first).total_seconds()
            )

            if not math.isfinite(seconds):
                raise ValueError(
                    "Invalid meteorological timestamp."
                )

            elapsed.append(seconds)

        if len(elapsed) > 1 and np.any(
            np.diff(elapsed) <= 0
        ):
            raise ValueError(
                "Meteorological timestamps must "
                "be strictly increasing."
            )

        interval = (
            float(elapsed[1] - elapsed[0])
            if len(elapsed) > 1
            else 0.0
        )

        self._time_info = {
            "count": len(data["times"]),
            "times": data["times"],
            "elapsed_seconds": elapsed,
            "interval_seconds": interval,
            "interval_hours": (
                interval / 3600.0
                if interval > 0
                else None
            )
        }

    def time_information(self):
        self.load()
        return self._time_info

    def time_index_at_elapsed(self, elapsed_seconds):
        info = self.time_information()
        times = np.asarray(
            info["elapsed_seconds"], dtype=float
        )

        target = float(elapsed_seconds)

        if not math.isfinite(target):
            raise ValueError(
                "Invalid meteorological elapsed time."
            )

        if target < times[0] or target > times[-1]:
            raise ValueError(
                "Time is outside meteorological coverage."
            )

        return int(np.argmin(np.abs(times - target)))

    @staticmethod
    def _axis_spacing(values):
        values = np.asarray(values, dtype=float)

        if values.size < 2:
            return None

        differences = np.abs(np.diff(values))
        differences = differences[
            np.isfinite(differences)
            & (differences > 0)
        ]

        return (
            float(np.median(differences))
            if differences.size
            else None
        )

    def horizontal_resolution(self, latitude=None):
        data = self.load()

        lat_step = self._axis_spacing(data["latitude"])
        lon_step = self._axis_spacing(data["longitude"])

        if lat_step is None or lon_step is None:
            raise ValueError(
                "Grid resolution cannot be determined."
            )

        latitude = float(
            latitude
            if latitude is not None
            else np.mean(data["latitude"])
        )

        lat_m = lat_step * 111320.0
        lon_m = (
            lon_step * 111320.0
            * abs(math.cos(math.radians(latitude)))
        )

        return float(min(lat_m, lon_m))

    def domain_radius(self, latitude=None, longitude=None):
        data = self.load()

        latitude = float(
            latitude
            if latitude is not None
            else np.mean(data["latitude"])
        )

        longitude = float(
            longitude
            if longitude is not None
            else np.mean(data["longitude"])
        )

        lat_min = float(np.min(data["latitude"]))
        lat_max = float(np.max(data["latitude"]))
        lon_min = float(np.min(data["longitude"]))
        lon_max = float(np.max(data["longitude"]))

        if not (
            lat_min <= latitude <= lat_max
            and lon_min <= longitude <= lon_max
        ):
            raise ValueError(
                "Location is outside meteorological domain."
            )

        lat_radius = min(
            latitude - lat_min,
            lat_max - latitude
        ) * 111320.0

        lon_radius = min(
            longitude - lon_min,
            lon_max - longitude
        ) * 111320.0 * abs(
            math.cos(math.radians(latitude))
        )

        return float(min(lat_radius, lon_radius))

    def _precipitation_variable(self):
        self.load()
        variables = self.dataset.variables

        names = (
            "tp", "total_precipitation", "precipitation",
            "precip", "pr", "rain", "rainfall",
            "precipitation_amount"
        )

        for name in names:
            if name in variables:
                return name

        for name, variable in variables.items():
            description = (
                f"{name} "
                f"{getattr(variable, 'long_name', '')} "
                f"{getattr(variable, 'standard_name', '')}"
            ).lower()

            if (
                "precip" in description
                or "rainfall" in description
            ):
                return name

        return None

    @staticmethod
    def _precipitation_to_meters(value, variable):
        units = str(
            getattr(variable, "units", "")
        ).lower().replace(" ", "").replace("**", "")

        if units in (
            "kgm-2", "kg/m2", "kgm^-2",
            "mm", "millimeter", "millimeters",
            "millimetre", "millimetres"
        ):
            return value / 1000.0

        if units in (
            "m", "meter", "meters",
            "metre", "metres"
        ):
            return value

        raise ValueError(
            f"Unsupported precipitation units: {units!r}"
        )

    def precipitation(
        self, time_index=0,
        latitude=None, longitude=None
    ):
        self.load()

        name = self._precipitation_variable()

        if name is None:
            return None

        variable = self.dataset.variables[name]
        values = self._array(variable)

        if values.size == 0:
            return None

        dimensions = list(variable.dimensions)
        indexes = {}

        for axis, dimension in enumerate(dimensions):
            key = dimension.lower()

            if key in (
                "valid_time", "time", "forecast_time"
            ):
                index = int(time_index)

                if not 0 <= index < values.shape[axis]:
                    raise ValueError(
                        "Invalid precipitation time index."
                    )

                indexes[axis] = index

            elif key in ("latitude", "lat"):
                target = float(
                    latitude
                    if latitude is not None
                    else np.mean(self.data["latitude"])
                )

                indexes[axis] = int(
                    np.abs(
                        self.data["latitude"] - target
                    ).argmin()
                )

            elif key in ("longitude", "lon"):
                target = float(
                    longitude
                    if longitude is not None
                    else np.mean(self.data["longitude"])
                )

                indexes[axis] = int(
                    np.abs(
                        self.data["longitude"] - target
                    ).argmin()
                )

            elif key in (
                "pressure_level", "isobaricinhpa", "level"
            ):
                indexes[axis] = 0

        if values.ndim == 0:
            result = self._valid(values.item())

        elif len(indexes) == values.ndim:
            result = self._valid(
                values[
                    tuple(
                        indexes[axis]
                        for axis in range(values.ndim)
                    )
                ]
            )

        elif values.ndim == 1:
            index = int(time_index)

            if not 0 <= index < values.shape[0]:
                raise ValueError(
                    "Invalid precipitation time index."
                )

            result = self._valid(values[index])

        else:
            return None

        if result is None:
            return None

        result = self._precipitation_to_meters(
            result, variable
        )

        return max(0.0, float(result))

    @staticmethod
    def _bounds(axis, target):
        axis = np.asarray(axis, dtype=float)
        target = float(target)

        if axis[0] > axis[-1]:
            ordered = axis[::-1]
            reversed_axis = True
        else:
            ordered = axis
            reversed_axis = False

        if not ordered[0] <= target <= ordered[-1]:
            raise ValueError(
                "Particle position is outside GFS domain."
            )

        index = int(np.clip(
            np.searchsorted(ordered, target) - 1,
            0,
            len(ordered) - 2
        ))

        fraction = (
            (target - ordered[index])
            / (ordered[index + 1] - ordered[index])
        )

        if reversed_axis:
            lower = len(axis) - 1 - index
            upper = len(axis) - 2 - index
        else:
            lower = index
            upper = index + 1

        return lower, upper, float(fraction)

    def bilinear_profile(
        self, latitude, longitude, time_index=0
    ):
        data = self.load()

        time_index = int(time_index)

        if not 0 <= time_index < len(data["times"]):
            raise ValueError(
                "Invalid meteorological time index."
            )

        i0, i1, fy = self._bounds(
            data["latitude"], latitude
        )

        j0, j1, fx = self._bounds(
            data["longitude"], longitude
        )

        def interpolate(name):
            values = data[name]

            if values is None:
                return None

            q00 = values[time_index, :, i0, j0]
            q01 = values[time_index, :, i0, j1]
            q10 = values[time_index, :, i1, j0]
            q11 = values[time_index, :, i1, j1]

            return (
                (1 - fy) * (1 - fx) * q00
                + (1 - fy) * fx * q01
                + fy * (1 - fx) * q10
                + fy * fx * q11
            )

        return {
            "pressure": data["pressure_level"],
            "u": interpolate("u"),
            "v": interpolate("v"),
            "temperature": interpolate("temperature"),
            "altitude": interpolate("altitude")
        }

    def nearest_grid(self, latitude, longitude):
        data = self.load()

        lat = float(latitude)
        lon = float(longitude)

        if not (
            np.min(data["latitude"])
            <= lat
            <= np.max(data["latitude"])
        ):
            raise ValueError(
                "Latitude is outside meteorological domain."
            )

        if not (
            np.min(data["longitude"])
            <= lon
            <= np.max(data["longitude"])
        ):
            raise ValueError(
                "Longitude is outside meteorological domain."
            )

        lat_index = int(
            np.abs(data["latitude"] - lat).argmin()
        )

        lon_index = int(
            np.abs(data["longitude"] - lon).argmin()
        )

        return {
            "latitude": float(
                data["latitude"][lat_index]
            ),
            "longitude": float(
                data["longitude"][lon_index]
            ),
            "latitude_index": lat_index,
            "longitude_index": lon_index
        }

    def atmospheric_profile(
        self, latitude, longitude, time_index=0
    ):
        data = self.load()
        time_index = int(time_index)

        if not 0 <= time_index < len(data["times"]):
            raise ValueError(
                "Invalid meteorological time index."
            )

        grid = self.nearest_grid(latitude, longitude)

        i = grid["latitude_index"]
        j = grid["longitude_index"]

        pressure = data["pressure_level"]
        u = data["u"][time_index, :, i, j]
        v = data["v"][time_index, :, i, j]

        temperature = (
            data["temperature"][time_index, :, i, j]
            if data["temperature"] is not None
            else None
        )

        geopotential = (
            data["geopotential"][time_index, :, i, j]
            if data["geopotential"] is not None
            else None
        )

        altitude = data["altitude"][
            time_index, :, i, j
        ]

        potential_temperature = None

        if temperature is not None:
            potential_temperature = (
                temperature
                * (1000.0 / np.maximum(
                    pressure, 0.001
                )) ** (287.05 / 1004.67)
            )

        profile = []

        for index in range(len(pressure)):
            level = {
                "pressure": float(pressure[index]),
                "u": float(u[index]),
                "v": float(v[index]),
                "altitude": float(altitude[index])
            }

            if temperature is not None:
                level["temperature"] = float(
                    temperature[index]
                )

            if geopotential is not None:
                level["geopotential"] = float(
                    geopotential[index]
                )

            if potential_temperature is not None:
                level["potential_temperature"] = float(
                    potential_temperature[index]
                )

            profile.append(level)

        return {
            "profile": profile,
            "pressure": pressure.tolist(),
            "u": u.tolist(),
            "v": v.tolist(),
            "temperature": (
                temperature.tolist()
                if temperature is not None
                else None
            ),
            "geopotential": (
                geopotential.tolist()
                if geopotential is not None
                else None
            ),
            "altitude": altitude.tolist(),
            "potential_temperature": (
                potential_temperature.tolist()
                if potential_temperature is not None
                else None
            )
        }

    
    def wind_components_at_altitude(self, latitude, longitude, altitude, time_index=0):
        profile = self.bilinear_profile(latitude, longitude, time_index)

        z = np.asarray(profile["altitude"], dtype=float)
        u = np.asarray(profile["u"], dtype=float)
        v = np.asarray(profile["v"], dtype=float)

        valid = np.isfinite(z) & np.isfinite(u) & np.isfinite(v)

        if np.count_nonzero(valid) < 2:
            raise ValueError("GFS wind profile has fewer than two valid levels.")

        z = z[valid]
        u = u[valid]
        v = v[valid]

        order = np.argsort(z)
        z = z[order]
        u = u[order]
        v = v[order]

        if np.any(np.diff(z) <= 0):
            raise ValueError("GFS altitude levels must be distinct.")

        target = float(altitude)

        if not math.isfinite(target):
            raise ValueError("Invalid particle altitude.")

        if target < z[0]:
            target = float(z[0])
        elif target > z[-1]:
            if target - z[-1] > 100.0:
                raise ValueError(
                    f"Particle altitude {target:.2f} m exceeds "
                    f"meteorological profile maximum {z[-1]:.2f} m by more than 100 m."
                )
            target = float(z[-1])

        return {
            "u": float(np.interp(target, z, u)),
            "v": float(np.interp(target, z, v)),
            "w": 0.0
        }

    def wind_components_at_elapsed(self, latitude, longitude, altitude, elapsed_seconds):
        times = self.time_information()["elapsed_seconds"]
        target = float(elapsed_seconds)

        if not math.isfinite(target):
            raise ValueError("Invalid meteorological elapsed time.")

        if target < times[0] or target > times[-1]:
            raise ValueError("Time is outside meteorological coverage.")

        upper = int(np.searchsorted(times, target, side="left"))

        if upper == 0:
            lower = upper = 0
        elif upper >= len(times):
            lower = upper = len(times) - 1
        elif times[upper] == target:
            lower = upper
        else:
            lower = upper - 1

        a = self.wind_components_at_altitude(
            latitude, longitude, altitude, lower
        )

        if lower == upper:
            return a

        b = self.wind_components_at_altitude(
            latitude, longitude, altitude, upper
        )

        fraction = (
            (target - times[lower]) /
            (times[upper] - times[lower])
        )

        return {
            "u": float(a["u"] + (b["u"] - a["u"]) * fraction),
            "v": float(a["v"] + (b["v"] - a["v"]) * fraction),
            "w": 0.0
        }

    def wind_at_altitude(
        self, latitude, longitude,
        altitude, time_index=0
    ):
        profile = self.bilinear_profile(
            latitude, longitude, time_index
        )

        z = np.asarray(
            profile["altitude"], dtype=float
        )

        u = np.asarray(
            profile["u"], dtype=float
        )

        v = np.asarray(
            profile["v"], dtype=float
        )

        pressure = np.asarray(
            profile["pressure"], dtype=float
        )

        temperature = profile["temperature"]

        valid = (
            np.isfinite(z)
            & np.isfinite(u)
            & np.isfinite(v)
        )

        if np.count_nonzero(valid) < 2:
            raise ValueError(
                "GFS wind profile has fewer "
                "than two valid levels."
            )

        z = z[valid]
        u = u[valid]
        v = v[valid]
        pressure = pressure[valid]

        if temperature is not None:
            temperature = np.asarray(
                temperature, dtype=float
            )[valid]

        order = np.argsort(z)

        z = z[order]
        u = u[order]
        v = v[order]
        pressure = pressure[order]

        if temperature is not None:
            temperature = temperature[order]

        if np.any(np.diff(z) <= 0):
            raise ValueError(
                "GFS altitude levels must be distinct."
            )

        target = float(altitude)

        if not math.isfinite(target):
            raise ValueError(
                "Invalid particle altitude."
            )

        if target < z[0]:
            target = float(z[0])

        elif target > z[-1]:
            if target - z[-1] > 100.0:
                raise ValueError(
                    f"Particle altitude {target:.2f} m "
                    f"exceeds meteorological profile "
                    f"maximum {z[-1]:.2f} m "
                    "by more than 100 m."
                )

            target = float(z[-1])

        wind_u = float(np.interp(target, z, u))
        wind_v = float(np.interp(target, z, v))

        brunt_vaisala = 0.011

        if temperature is not None:
            theta = (
                temperature
                * (1000.0 / np.maximum(
                    pressure, 0.001
                )) ** (287.05 / 1004.67)
            )

            valid_theta = (
                np.isfinite(theta)
                & (theta > 0)
            )

            if np.count_nonzero(valid_theta) >= 2:
                tz = z[valid_theta]
                tt = theta[valid_theta]

                theta_target = float(
                    np.interp(target, tz, tt)
                )

                gradient = np.gradient(tt, tz)

                gradient_target = float(
                    np.interp(
                        target, tz, gradient
                    )
                )

                if theta_target > 0:
                    n2 = (
                        9.81 / theta_target
                        * gradient_target
                    )

                    brunt_vaisala = max(
                        math.sqrt(max(n2, 0.0)),
                        0.011
                    )

        return {
            "u": wind_u,
            "v": wind_v,
            "wind_speed": math.hypot(
                wind_u, wind_v
            ),
            "brunt_vaisala": float(
                brunt_vaisala
            ),
            "altitude": target,
            "time_index": int(time_index)
        }

    def wind_at_elapsed(
        self, latitude, longitude,
        altitude, elapsed_seconds
    ):
        info = self.time_information()

        times = np.asarray(
            info["elapsed_seconds"], dtype=float
        )

        target = float(elapsed_seconds)

        if not math.isfinite(target):
            raise ValueError(
                "Invalid meteorological elapsed time."
            )

        if target < times[0] or target > times[-1]:
            raise ValueError(
                "Time is outside meteorological coverage."
            )

        upper = int(
            np.searchsorted(
                times, target, side="left"
            )
        )

        if upper == 0:
            lower = upper = 0

        elif upper >= len(times):
            lower = upper = len(times) - 1

        elif times[upper] == target:
            lower = upper

        else:
            lower = upper - 1

        a = self.wind_at_altitude(
            latitude, longitude,
            altitude, lower
        )

        if lower == upper:
            return {
                **a,
                "elapsed_seconds": target
            }

        b = self.wind_at_altitude(
            latitude, longitude,
            altitude, upper
        )

        fraction = (
            (target - times[lower])
            / (times[upper] - times[lower])
        )

        u = (
            a["u"]
            + (b["u"] - a["u"]) * fraction
        )

        v = (
            a["v"]
            + (b["v"] - a["v"]) * fraction
        )

        n = (
            a["brunt_vaisala"]
            + (
                b["brunt_vaisala"]
                - a["brunt_vaisala"]
            ) * fraction
        )

        return {
            "u": float(u),
            "v": float(v),
            "wind_speed": math.hypot(u, v),
            "brunt_vaisala": float(n),
            "altitude": float(altitude),
            "elapsed_seconds": target
        }

    def dataset_information(self):
        data = self.load()
        info = self.time_information()

        return {
            "file": str(self.file_path),
            "latitude_count": len(data["latitude"]),
            "longitude_count": len(data["longitude"]),
            "pressure_level_count": len(
                data["pressure_level"]
            ),
            "time_count": len(data["times"]),
            "latitude_min": float(
                np.min(data["latitude"])
            ),
            "latitude_max": float(
                np.max(data["latitude"])
            ),
            "longitude_min": float(
                np.min(data["longitude"])
            ),
            "longitude_max": float(
                np.max(data["longitude"])
            ),
            "pressure_min": float(
                np.min(data["pressure_level"])
            ),
            "pressure_max": float(
                np.max(data["pressure_level"])
            ),
            "interval_hours": info["interval_hours"]
        }

    def close(self):
        if self.dataset is not None:
            try:
                self.dataset.close()
            finally:
                self.dataset = None

        self.data = None
        self._time_info = None
