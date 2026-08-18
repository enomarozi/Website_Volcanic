from pathlib import Path
import math

import numpy as np
from netCDF4 import Dataset, num2date


class MeteorologyService:

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.dataset = Dataset(
            str(self.dataset_path),
            mode="r"
        )

        variables = self.dataset.variables

        # ---------------------------------------------------------
        # REQUIRED VARIABLES
        # ---------------------------------------------------------

        required_variables = [
            "latitude",
            "longitude",
            "pressure_level",
            "valid_time",
            "u",
            "v"
        ]

        for variable in required_variables:

            if variable not in variables:

                raise ValueError(
                    f"Dataset does not contain "
                    f"{variable} variable."
                )

        # ---------------------------------------------------------
        # LOAD COORDINATES
        # ---------------------------------------------------------

        self.latitude = np.asarray(
            variables["latitude"][:],
            dtype=float
        )

        self.longitude = np.asarray(
            variables["longitude"][:],
            dtype=float
        )

        self.pressure_level = np.asarray(
            variables["pressure_level"][:],
            dtype=float
        )

        # ---------------------------------------------------------
        # LOAD WIND
        # ---------------------------------------------------------

        self.u = np.asarray(
            variables["u"][:],
            dtype=float
        )

        self.v = np.asarray(
            variables["v"][:],
            dtype=float
        )

        # ---------------------------------------------------------
        # TEMPERATURE
        # ---------------------------------------------------------

        self.temperature = None

        if "temperature" in variables:

            self.temperature = np.asarray(
                variables["temperature"][:],
                dtype=float
            )

        elif "t" in variables:

            self.temperature = np.asarray(
                variables["t"][:],
                dtype=float
            )

        # ---------------------------------------------------------
        # GEOPOTENTIAL
        # ---------------------------------------------------------

        self.geopotential = None

        if "geopotential" in variables:

            self.geopotential = np.asarray(
                variables["geopotential"][:],
                dtype=float
            )

        elif "z" in variables:

            self.geopotential = np.asarray(
                variables["z"][:],
                dtype=float
            )

        # ---------------------------------------------------------
        # TIME
        # ---------------------------------------------------------

        self.valid_time = self._load_times(
            variables["valid_time"]
        )

        # ---------------------------------------------------------
        # VALIDATE DATA DIMENSIONS
        # ---------------------------------------------------------

        self._validate_dimensions()

    # =============================================================
    # TIME
    # =============================================================

    def _load_times(self, variable):

        values = np.asarray(
            variable[:]
        )

        # netCDF4 normally stores valid_time
        # as numeric values with units.
        if hasattr(variable, "units"):

            calendar = getattr(
                variable,
                "calendar",
                "standard"
            )

            try:

                converted = num2date(
                    values,
                    units=variable.units,
                    calendar=calendar,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True
                )

                return list(converted)

            except Exception:

                pass

        # Fallback if already datetime-like.
        result = []

        for value in values:

            try:
                result.append(
                    value.item()
                )

            except Exception:

                result.append(
                    value
                )

        return result

    def _validate_dimensions(self):

        if self.u.ndim != 4:

            raise ValueError(
                "Variable 'u' must have "
                "4 dimensions: "
                "(time, pressure, latitude, longitude). "
                f"Received shape: {self.u.shape}"
            )

        if self.v.ndim != 4:

            raise ValueError(
                "Variable 'v' must have "
                "4 dimensions: "
                "(time, pressure, latitude, longitude). "
                f"Received shape: {self.v.shape}"
            )

        expected_shape = (
            len(self.valid_time),
            len(self.pressure_level),
            len(self.latitude),
            len(self.longitude)
        )

        if self.u.shape != expected_shape:

            raise ValueError(
                "Unexpected shape for variable 'u'. "
                f"Expected {expected_shape}, "
                f"received {self.u.shape}"
            )

        if self.v.shape != expected_shape:

            raise ValueError(
                "Unexpected shape for variable 'v'. "
                f"Expected {expected_shape}, "
                f"received {self.v.shape}"
            )

        if self.temperature is not None:

            if self.temperature.shape != expected_shape:

                raise ValueError(
                    "Unexpected shape for temperature. "
                    f"Expected {expected_shape}, "
                    f"received {self.temperature.shape}"
                )

        if self.geopotential is not None:

            if self.geopotential.shape != expected_shape:

                raise ValueError(
                    "Unexpected shape for geopotential. "
                    f"Expected {expected_shape}, "
                    f"received {self.geopotential.shape}"
                )

    # =============================================================
    # GRID
    # =============================================================

    def nearest_grid(
        self,
        latitude: float,
        longitude: float
    ):

        latitude = float(latitude)
        longitude = float(longitude)

        latitude_index = int(
            np.abs(
                self.latitude - latitude
            ).argmin()
        )

        longitude_index = int(
            np.abs(
                self.longitude - longitude
            ).argmin()
        )

        return {

            "latitude": float(
                self.latitude[
                    latitude_index
                ]
            ),

            "longitude": float(
                self.longitude[
                    longitude_index
                ]
            ),

            "latitude_index":
                latitude_index,

            "longitude_index":
                longitude_index,

            "requested_latitude":
                latitude,

            "requested_longitude":
                longitude
        }

    # =============================================================
    # TIME INFORMATION
    # =============================================================

    def time_information(self):

        times = list(
            self.valid_time
        )

        if not times:

            return {
                "count": 0,
                "times": [],
                "elapsed_seconds": [],
                "interval_seconds": 0.0,
                "interval_hours": None
            }

        elapsed_seconds = []

        first_time = times[0]

        for index, current_time in enumerate(
            times
        ):

            try:

                delta = (
                    current_time
                    - first_time
                )

                elapsed = (
                    delta.total_seconds()
                )

            except Exception:

                # Fallback to 6-hour ERA5 interval
                elapsed = (
                    index * 21600.0
                )

            elapsed_seconds.append(
                float(elapsed)
            )

        if len(elapsed_seconds) > 1:

            differences = np.diff(
                elapsed_seconds
            )

            positive_differences = (
                differences[
                    differences > 0
                ]
            )

            if len(
                positive_differences
            ) > 0:

                interval_seconds = float(
                    positive_differences[0]
                )

            else:

                interval_seconds = 0.0

        else:

            interval_seconds = 0.0

        return {

            "count":
                len(times),

            "times":
                times,

            "elapsed_seconds":
                elapsed_seconds,

            "interval_seconds":
                interval_seconds,

            "interval_hours":
                (
                    interval_seconds / 3600.0
                    if interval_seconds > 0
                    else None
                )
        }

    # =============================================================
    # TIME INDEX
    # =============================================================

    def time_index_at_elapsed(
        self,
        elapsed_seconds: float
    ):

        information = (
            self.time_information()
        )

        elapsed = np.asarray(
            information[
                "elapsed_seconds"
            ],
            dtype=float
        )

        if len(elapsed) == 0:

            return 0

        index = int(
            np.abs(
                elapsed
                - float(elapsed_seconds)
            ).argmin()
        )

        return max(
            0,
            min(
                index,
                len(elapsed) - 1
            )
        )

    # =============================================================
    # TIME INDEX AT DATETIME
    # =============================================================

    def time_index_at_time(
        self,
        target_time
    ):

        if not self.valid_time:

            return 0

        differences = []

        for current_time in self.valid_time:

            try:

                difference = abs(
                    (
                        current_time
                        - target_time
                    ).total_seconds()
                )

            except Exception:

                difference = float(
                    "inf"
                )

            differences.append(
                difference
            )

        return int(
            np.argmin(
                differences
            )
        )

    # =============================================================
    # PRESSURE INDEX
    # =============================================================

    def nearest_pressure_level(
        self,
        pressure_level: float
    ):

        pressure_level = float(
            pressure_level
        )

        index = int(
            np.abs(
                self.pressure_level
                - pressure_level
            ).argmin()
        )

        return {

            "pressure_level":
                float(
                    self.pressure_level[
                        index
                    ]
                ),

            "pressure_index":
                index,

            "requested_pressure":
                pressure_level
        }

    # =============================================================
    # TEMPERATURE
    # =============================================================

    def _temperature(
        self,
        time_index: int,
        pressure_index: int,
        latitude_index: int,
        longitude_index: int
    ):

        if self.temperature is not None:

            value = self.temperature[
                time_index,
                pressure_index,
                latitude_index,
                longitude_index
            ]

            return float(
                value
            )

        # Standard atmosphere fallback.
        pressure = float(
            self.pressure_level[
                pressure_index
            ]
        )

        return float(
            288.15
            * (
                pressure / 1013.25
            ) ** 0.286
        )

    # =============================================================
    # ALTITUDE
    # =============================================================

    def _altitude(
        self,
        time_index: int,
        pressure_index: int,
        latitude_index: int,
        longitude_index: int
    ):

        if self.geopotential is not None:

            geopotential = float(
                self.geopotential[
                    time_index,
                    pressure_index,
                    latitude_index,
                    longitude_index
                ]
            )

            return float(
                geopotential / 9.80665
            )

        # Hypsometric / standard atmosphere fallback.
        pressure = float(
            self.pressure_level[
                pressure_index
            ]
        )

        return float(
            44330.0
            * (
                1.0
                - (
                    pressure / 1013.25
                ) ** 0.1903
            )
        )

    # =============================================================
    # WIND DIRECTION
    # =============================================================

    @staticmethod
    def _wind_direction(
        u: float,
        v: float
    ):

        direction = (
            math.degrees(
                math.atan2(
                    -u,
                    -v
                )
            )
            + 360.0
        ) % 360.0

        return float(
            direction
        )

    # =============================================================
    # WIND SPEED
    # =============================================================

    @staticmethod
    def _wind_speed(
        u: float,
        v: float
    ):

        return float(
            math.sqrt(
                u ** 2
                + v ** 2
            )
        )

    # =============================================================
    # ATMOSPHERIC PROFILE
    # =============================================================

    def atmospheric_profile(
        self,
        latitude: float,
        longitude: float,
        time_index: int = 0
    ):

        grid = self.nearest_grid(
            latitude=latitude,
            longitude=longitude
        )

        latitude_index = (
            grid[
                "latitude_index"
            ]
        )

        longitude_index = (
            grid[
                "longitude_index"
            ]
        )

        time_index = max(
            0,
            min(
                int(time_index),
                len(self.valid_time) - 1
            )
        )

        profile = []

        for pressure_index, pressure in enumerate(
            self.pressure_level
        ):

            pressure = float(
                pressure
            )

            altitude = self._altitude(
                time_index=time_index,
                pressure_index=pressure_index,
                latitude_index=latitude_index,
                longitude_index=longitude_index
            )

            temperature = self._temperature(
                time_index=time_index,
                pressure_index=pressure_index,
                latitude_index=latitude_index,
                longitude_index=longitude_index
            )

            u = float(
                self.u[
                    time_index,
                    pressure_index,
                    latitude_index,
                    longitude_index
                ]
            )

            v = float(
                self.v[
                    time_index,
                    pressure_index,
                    latitude_index,
                    longitude_index
                ]
            )

            pressure_pa = (
                pressure * 100.0
            )

            potential_temperature = (
                temperature
                * (
                    100000.0
                    / pressure_pa
                ) ** 0.286
            )

            wind_speed = self._wind_speed(
                u,
                v
            )

            wind_direction = (
                self._wind_direction(
                    u,
                    v
                )
            )

            profile.append({

                "pressure_level":
                    pressure,

                "altitude":
                    altitude,

                "u":
                    u,

                "v":
                    v,

                "wind_speed":
                    wind_speed,

                "wind_direction":
                    wind_direction,

                "temperature":
                    temperature,

                "potential_temperature":
                    float(
                        potential_temperature
                    ),

                "brunt_vaisala":
                    0.0
            })

        # ---------------------------------------------------------
        # Sort profile by altitude.
        #
        # This is important because ERA5 pressure levels can be
        # ordered from high pressure to low pressure, while altitude
        # interpolation needs increasing altitude.
        # ---------------------------------------------------------

        profile.sort(
            key=lambda item:
                item["altitude"]
        )

        # ---------------------------------------------------------
        # BRUNT-VÄISÄLÄ FREQUENCY
        # ---------------------------------------------------------

        for index in range(
            len(profile)
        ):

            if index == 0:

                profile[index][
                    "brunt_vaisala"
                ] = 0.0

                continue

            current = profile[
                index
            ]

            previous = profile[
                index - 1
            ]

            dz = (
                current[
                    "altitude"
                ]
                - previous[
                    "altitude"
                ]
            )

            dtheta = (
                current[
                    "potential_temperature"
                ]
                - previous[
                    "potential_temperature"
                ]
            )

            if abs(dz) <= 1e-12:

                current[
                    "brunt_vaisala"
                ] = 0.0

                continue

            dtheta_dz = (
                dtheta / dz
            )

            value = (
                9.80665
                / max(
                    current[
                        "potential_temperature"
                    ],
                    1e-12
                )
            ) * dtheta_dz

            current[
                "brunt_vaisala"
            ] = math.sqrt(
                max(
                    0.0,
                    value
                )
            )

        return {

            "latitude":
                float(latitude),

            "longitude":
                float(longitude),

            "grid":
                grid,

            "time_index":
                time_index,

            "time":
                (
                    str(
                        self.valid_time[
                            time_index
                        ]
                    )
                    if self.valid_time
                    else None
                ),

            "profile":
                profile
        }

    # =============================================================
    # INTERPOLATE PROFILE
    # =============================================================

    def _interpolate_profile(
        self,
        profile,
        altitude: float
    ):

        if not profile:

            return {

                "u": 0.0,
                "v": 0.0,
                "wind_speed": 0.0,
                "wind_direction": 0.0,
                "altitude": float(
                    altitude
                ),
                "pressure_level": None,
                "temperature": None,
                "potential_temperature": None,
                "brunt_vaisala": 0.0
            }

        altitude = float(
            altitude
        )

        # ---------------------------------------------------------
        # BELOW LOWEST LEVEL
        # ---------------------------------------------------------

        if altitude <= profile[0][
            "altitude"
        ]:

            selected = profile[0]

            return {
                key: value
                for key, value in selected.items()
            }

        # ---------------------------------------------------------
        # ABOVE HIGHEST LEVEL
        # ---------------------------------------------------------

        if altitude >= profile[-1][
            "altitude"
        ]:

            selected = profile[-1]

            return {
                key: value
                for key, value in selected.items()
            }

        # ---------------------------------------------------------
        # FIND BRACKETING LEVELS
        # ---------------------------------------------------------

        lower = profile[0]
        upper = profile[-1]

        for index in range(
            len(profile) - 1
        ):

            first = profile[
                index
            ]

            second = profile[
                index + 1
            ]

            if (
                first["altitude"]
                <= altitude
                <= second["altitude"]
            ):

                lower = first
                upper = second

                break

        z1 = float(
            lower["altitude"]
        )

        z2 = float(
            upper["altitude"]
        )

        if abs(z2 - z1) <= 1e-12:

            factor = 0.0

        else:

            factor = (
                altitude - z1
            ) / (
                z2 - z1
            )

        factor = max(
            0.0,
            min(
                1.0,
                factor
            )
        )

        # ---------------------------------------------------------
        # LINEAR INTERPOLATION
        # ---------------------------------------------------------

        u = (
            lower["u"]
            + factor
            * (
                upper["u"]
                - lower["u"]
            )
        )

        v = (
            lower["v"]
            + factor
            * (
                upper["v"]
                - lower["v"]
            )
        )

        temperature = None

        if (
            lower["temperature"]
            is not None
            and upper["temperature"]
            is not None
        ):

            temperature = (
                lower["temperature"]
                + factor
                * (
                    upper["temperature"]
                    - lower["temperature"]
                )
            )

        potential_temperature = None

        if (
            lower[
                "potential_temperature"
            ]
            is not None
            and upper[
                "potential_temperature"
            ]
            is not None
        ):

            potential_temperature = (
                lower[
                    "potential_temperature"
                ]
                + factor
                * (
                    upper[
                        "potential_temperature"
                    ]
                    - lower[
                        "potential_temperature"
                    ]
                )
            )

        brunt_vaisala = (
            lower[
                "brunt_vaisala"
            ]
            + factor
            * (
                upper[
                    "brunt_vaisala"
                ]
                - lower[
                    "brunt_vaisala"
                ]
            )
        )

        wind_speed = self._wind_speed(
            u,
            v
        )

        wind_direction = (
            self._wind_direction(
                u,
                v
            )
        )

        pressure_level = (
            lower[
                "pressure_level"
            ]
            + factor
            * (
                upper[
                    "pressure_level"
                ]
                - lower[
                    "pressure_level"
                ]
            )
        )

        return {

            "u":
                float(u),

            "v":
                float(v),

            "wind_speed":
                float(wind_speed),

            "wind_direction":
                float(wind_direction),

            "altitude":
                float(altitude),

            "pressure_level":
                float(pressure_level),

            "temperature":
                (
                    float(temperature)
                    if temperature is not None
                    else None
                ),

            "potential_temperature":
                (
                    float(
                        potential_temperature
                    )
                    if potential_temperature is not None
                    else None
                ),

            "brunt_vaisala":
                float(
                    brunt_vaisala
                )
        }

    # =============================================================
    # WIND AT ALTITUDE
    # =============================================================

    def wind_at_altitude(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        time_index: int = 0
    ):

        atmospheric = (
            self.atmospheric_profile(
                latitude=latitude,
                longitude=longitude,
                time_index=time_index
            )
        )

        result = (
            self._interpolate_profile(
                profile=atmospheric[
                    "profile"
                ],
                altitude=altitude
            )
        )

        result.update({

            "latitude":
                float(latitude),

            "longitude":
                float(longitude),

            "time_index":
                int(
                    atmospheric[
                        "time_index"
                    ]
                ),

            "time":
                atmospheric.get(
                    "time"
                )
        })

        return result

    # =============================================================
    # WIND AT PRESSURE LEVEL
    # =============================================================

    def wind_at_pressure_level(
        self,
        latitude: float,
        longitude: float,
        pressure_level: float,
        time_index: int = 0
    ):

        grid = self.nearest_grid(
            latitude=latitude,
            longitude=longitude
        )

        pressure = (
            self.nearest_pressure_level(
                pressure_level
            )
        )

        latitude_index = (
            grid[
                "latitude_index"
            ]
        )

        longitude_index = (
            grid[
                "longitude_index"
            ]
        )

        pressure_index = (
            pressure[
                "pressure_index"
            ]
        )

        time_index = max(
            0,
            min(
                int(time_index),
                len(self.valid_time) - 1
            )
        )

        u = float(
            self.u[
                time_index,
                pressure_index,
                latitude_index,
                longitude_index
            ]
        )

        v = float(
            self.v[
                time_index,
                pressure_index,
                latitude_index,
                longitude_index
            ]
        )

        return {

            "latitude":
                float(latitude),

            "longitude":
                float(longitude),

            "pressure_level":
                float(
                    self.pressure_level[
                        pressure_index
                    ]
                ),

            "altitude":
                self._altitude(
                    time_index=time_index,
                    pressure_index=pressure_index,
                    latitude_index=latitude_index,
                    longitude_index=longitude_index
                ),

            "u":
                u,

            "v":
                v,

            "wind_speed":
                self._wind_speed(
                    u,
                    v
                ),

            "wind_direction":
                self._wind_direction(
                    u,
                    v
                ),

            "time_index":
                time_index
        }

    # =============================================================
    # FULL DATASET INFORMATION
    # =============================================================

    def dataset_information(self):

        information = (
            self.time_information()
        )

        return {

            "dataset":
                str(
                    self.dataset_path
                ),

            "latitude_count":
                len(
                    self.latitude
                ),

            "longitude_count":
                len(
                    self.longitude
                ),

            "pressure_level_count":
                len(
                    self.pressure_level
                ),

            "time_count":
                information[
                    "count"
                ],

            "latitude_min":
                float(
                    np.min(
                        self.latitude
                    )
                ),

            "latitude_max":
                float(
                    np.max(
                        self.latitude
                    )
                ),

            "longitude_min":
                float(
                    np.min(
                        self.longitude
                    )
                ),

            "longitude_max":
                float(
                    np.max(
                        self.longitude
                    )
                ),

            "pressure_levels":
                [
                    float(
                        value
                    )
                    for value
                    in self.pressure_level
                ],

            "times":
                [
                    str(
                        value
                    )
                    for value
                    in information[
                        "times"
                    ]
                ],

            "interval_seconds":
                information[
                    "interval_seconds"
                ],

            "interval_hours":
                information[
                    "interval_hours"
                ],

            "has_temperature":
                self.temperature
                is not None,

            "has_geopotential":
                self.geopotential
                is not None
        }

    # =============================================================
    # CLOSE DATASET
    # =============================================================

    def close(self):

        if self.dataset is not None:

            self.dataset.close()

            self.dataset = None

    # =============================================================
    # CONTEXT MANAGER
    # =============================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        self.close()