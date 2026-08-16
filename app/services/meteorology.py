import numpy as np
from pathlib import Path
from netCDF4 import Dataset, num2date


class MeteorologyService:

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"NetCDF file not found: {self.file_path}"
            )

        with Dataset(
            self.file_path,
            mode="r"
        ) as ds:

            required = [
                "valid_time",
                "pressure_level",
                "latitude",
                "longitude",
                "u",
                "v",
                "t",
                "z"
            ]

            missing = [
                name
                for name in required
                if name not in ds.variables
            ]

            if missing:
                raise ValueError(
                    "Required meteorological variables are missing: "
                    f"{', '.join(missing)}"
                )

            time_variable = (
                ds.variables["valid_time"]
            )

            raw_time = np.asarray(
                time_variable[:]
            )

            if hasattr(
                time_variable,
                "units"
            ):

                calendar = getattr(
                    time_variable,
                    "calendar",
                    "standard"
                )

                converted_time = num2date(
                    raw_time,
                    units=time_variable.units,
                    calendar=calendar
                )

                time = np.asarray(
                    converted_time
                )

            else:
                time = raw_time

            return {
                "time": time,
                "pressure_level": np.asarray(
                    ds.variables[
                        "pressure_level"
                    ][:]
                ),
                "latitude": np.asarray(
                    ds.variables[
                        "latitude"
                    ][:]
                ),
                "longitude": np.asarray(
                    ds.variables[
                        "longitude"
                    ][:]
                ),
                "u": np.asarray(
                    ds.variables["u"][:],
                    dtype=float
                ),
                "v": np.asarray(
                    ds.variables["v"][:],
                    dtype=float
                ),
                "temperature": np.asarray(
                    ds.variables["t"][:],
                    dtype=float
                ),
                "geopotential": np.asarray(
                    ds.variables["z"][:],
                    dtype=float
                )
            }

    def nearest_grid(
        self,
        latitude: float,
        longitude: float
    ):
        data = self.load()

        latitudes = data["latitude"]
        longitudes = data["longitude"]

        lat_index = np.abs(
            latitudes - latitude
        ).argmin()

        lon_index = np.abs(
            longitudes - longitude
        ).argmin()

        return {
            "latitude": float(
                latitudes[lat_index]
            ),
            "longitude": float(
                longitudes[lon_index]
            ),
            "latitude_index": int(
                lat_index
            ),
            "longitude_index": int(
                lon_index
            )
        }

    def pressure_to_pa(
        self,
        pressure_hpa
    ):
        return np.asarray(
            pressure_hpa,
            dtype=float
        ) * 100.0

    def potential_temperature(
        self,
        temperature,
        pressure_hpa
    ):
        temperature = np.asarray(
            temperature,
            dtype=float
        )

        pressure_hpa = np.asarray(
            pressure_hpa,
            dtype=float
        )

        p0 = 1000.0
        gamma = 1.41

        exponent = (
            gamma - 1.0
        ) / gamma

        return (
            temperature
            * (p0 / pressure_hpa)
            ** exponent
        )

    def geopotential_to_altitude(
        self,
        geopotential
    ):
        geopotential = np.asarray(
            geopotential,
            dtype=float
        )

        return geopotential / 9.81

    def brunt_vaisala_frequency(
        self,
        potential_temperature,
        altitude
    ):
        theta = np.asarray(
            potential_temperature,
            dtype=float
        )

        altitude = np.asarray(
            altitude,
            dtype=float
        )

        if len(theta) < 2:
            raise ValueError(
                "At least two atmospheric levels are required."
            )

        dtheta_dz = np.gradient(
            theta,
            altitude
        )

        n_squared = (
            9.81
            / np.maximum(
                theta,
                1e-12
            )
        ) * dtheta_dz

        n_squared = np.maximum(
            n_squared,
            0.0
        )

        return np.sqrt(
            n_squared
        )

    def atmospheric_profile(
        self,
        latitude: float,
        longitude: float,
        time_index: int
    ):
        data = self.load()

        latitudes = data["latitude"]
        longitudes = data["longitude"]
        pressure_levels = (
            data["pressure_level"]
        )

        lat_index = np.abs(
            latitudes - latitude
        ).argmin()

        lon_index = np.abs(
            longitudes - longitude
        ).argmin()

        if (
            time_index < 0
            or time_index >= len(
                data["time"]
            )
        ):
            raise IndexError(
                "Invalid time_index."
            )

        pressures = (
            self.pressure_to_pa(
                pressure_levels
            )
        )

        temperatures = data[
            "temperature"
        ][
            time_index,
            :,
            lat_index,
            lon_index
        ]

        geopotential = data[
            "geopotential"
        ][
            time_index,
            :,
            lat_index,
            lon_index
        ]

        altitude = (
            self.geopotential_to_altitude(
                geopotential
            )
        )

        potential_temperatures = (
            self.potential_temperature(
                temperatures,
                pressure_levels
            )
        )

        brunt_vaisala = (
            self.brunt_vaisala_frequency(
                potential_temperatures,
                altitude
            )
        )

        profile = []

        for index, pressure in enumerate(
            pressure_levels
        ):
            profile.append({
                "pressure_level": float(
                    pressure
                ),
                "pressure_pa": float(
                    pressures[index]
                ),
                "altitude": float(
                    altitude[index]
                ),
                "u": float(
                    data["u"][
                        time_index,
                        index,
                        lat_index,
                        lon_index
                    ]
                ),
                "v": float(
                    data["v"][
                        time_index,
                        index,
                        lat_index,
                        lon_index
                    ]
                ),
                "temperature": float(
                    temperatures[index]
                ),
                "potential_temperature": float(
                    potential_temperatures[index]
                ),
                "brunt_vaisala": float(
                    brunt_vaisala[index]
                )
            })

        return {
            "latitude": float(
                latitudes[lat_index]
            ),
            "longitude": float(
                longitudes[lon_index]
            ),
            "time_index": int(
                time_index
            ),
            "time": data[
                "time"
            ][time_index],
            "profile": profile
        }

    def wind_at_altitude(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        time_index: int
    ):
        profile = (
            self.atmospheric_profile(
                latitude,
                longitude,
                time_index
            )
        )

        levels = profile[
            "profile"
        ]

        altitudes = np.asarray([
            item["altitude"]
            for item in levels
        ])

        u = np.asarray([
            item["u"]
            for item in levels
        ])

        v = np.asarray([
            item["v"]
            for item in levels
        ])

        order = np.argsort(
            altitudes
        )

        altitudes = altitudes[
            order
        ]

        u = u[
            order
        ]

        v = v[
            order
        ]

        altitude = float(
            np.clip(
                altitude,
                altitudes.min(),
                altitudes.max()
            )
        )

        return {
            "u": float(
                np.interp(
                    altitude,
                    altitudes,
                    u
                )
            ),
            "v": float(
                np.interp(
                    altitude,
                    altitudes,
                    v
                )
            ),
            "altitude": altitude,
            "time_index": int(
                time_index
            ),
            "time": profile[
                "time"
            ]
        }

    def wind_profile(
        self,
        latitude: float,
        longitude: float,
        time_index: int
    ):
        profile = (
            self.atmospheric_profile(
                latitude,
                longitude,
                time_index
            )
        )

        return {
            "latitude": profile[
                "latitude"
            ],
            "longitude": profile[
                "longitude"
            ],
            "time_index": profile[
                "time_index"
            ],
            "time": profile[
                "time"
            ],
            "profile": [
                {
                    "pressure_level": item[
                        "pressure_level"
                    ],
                    "altitude": item[
                        "altitude"
                    ],
                    "u": item["u"],
                    "v": item["v"]
                }
                for item in profile[
                    "profile"
                ]
            ]
        }

    def time_information(self):
        data = self.load()

        times = data[
            "time"
        ]

        if len(times) == 0:
            raise ValueError(
                "Meteorological time data is empty."
            )

        if len(times) == 1:
            return {
                "count": 1,
                "interval_seconds": None,
                "interval_hours": None,
                "intervals_seconds": [],
                "times": times
            }

        intervals_seconds = []

        for index in range(
            1,
            len(times)
        ):
            delta = (
                times[index]
                - times[index - 1]
            )

            seconds = (
                delta.total_seconds()
            )

            if seconds <= 0:
                raise ValueError(
                    "Meteorological times must be "
                    "strictly increasing."
                )

            intervals_seconds.append(
                float(seconds)
            )

        return {
            "count": len(times),
            "interval_seconds": (
                intervals_seconds[0]
            ),
            "interval_hours": (
                intervals_seconds[0]
                / 3600.0
            ),
            "intervals_seconds": (
                intervals_seconds
            ),
            "times": times
        }

    def find_time_index(
        self,
        target_time
    ):
        data = self.load()

        times = data[
            "time"
        ]

        if len(times) == 0:
            raise ValueError(
                "Meteorological time data is empty."
            )

        differences = []

        for time in times:
            differences.append(
                abs(
                    (
                        time
                        - target_time
                    ).total_seconds()
                )
            )

        return int(
            np.argmin(
                differences
            )
        )

    def simulation_time_range(
        self,
        time_index: int,
        duration: float
    ):
        information = (
            self.time_information()
        )

        times = information[
            "times"
        ]

        if (
            time_index < 0
            or time_index >= len(times)
        ):
            raise IndexError(
                "Invalid simulation start time_index."
            )

        start_time = times[
            time_index
        ]

        elapsed = 0.0
        current_index = time_index

        while (
            elapsed < duration
        ):

            if (
                current_index
                >= len(times) - 1
            ):
                raise ValueError(
                    "Simulation duration exceeds "
                    "available meteorological data."
                )

            interval = information[
                "intervals_seconds"
            ][current_index]

            elapsed += interval
            current_index += 1

        return {
            "start_time": start_time,
            "end_time": (
                start_time
                + (
                    times[
                        current_index
                    ]
                    - times[
                        time_index
                    ]
                )
            ),
            "start_index": time_index,
            "end_index": current_index,
            "duration_seconds": elapsed
        }