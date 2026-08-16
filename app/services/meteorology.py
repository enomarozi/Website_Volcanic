import numpy as np
from pathlib import Path
from netCDF4 import Dataset


class MeteorologyService:

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"NetCDF file not found: {self.file_path}"
            )

        with Dataset(self.file_path, mode="r") as ds:
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
                    + ", ".join(missing)
                )

            time = np.asarray(
                ds.variables["valid_time"][:]
            )

            return {
                "time": time,
                "pressure_level": np.asarray(
                    ds.variables["pressure_level"][:]
                ),
                "latitude": np.asarray(
                    ds.variables["latitude"][:]
                ),
                "longitude": np.asarray(
                    ds.variables["longitude"][:]
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

        latitude_index = np.abs(
            latitudes - latitude
        ).argmin()

        longitude_index = np.abs(
            longitudes - longitude
        ).argmin()

        return {
            "latitude": float(
                latitudes[latitude_index]
            ),
            "longitude": float(
                longitudes[longitude_index]
            ),
            "latitude_index": int(
                latitude_index
            ),
            "longitude_index": int(
                longitude_index
            )
        }

    def pressure_to_pa(
        self,
        pressure_hpa
    ):
        return (
            np.asarray(
                pressure_hpa,
                dtype=float
            ) * 100.0
        )

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
            (gamma - 1.0) / gamma
        )

        return (
            temperature
            * (
                p0 / pressure_hpa
            ) ** exponent
        )

    def geopotential_to_altitude(
        self,
        geopotential
    ):
        return (
            np.asarray(
                geopotential,
                dtype=float
            ) / 9.81
        )

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

        order = np.argsort(
            altitude
        )

        altitude = altitude[order]
        theta = theta[order]

        unique_altitude, unique_index = np.unique(
            altitude,
            return_index=True
        )

        altitude = unique_altitude
        theta = theta[unique_index]

        if len(theta) < 2:
            raise ValueError(
                "At least two unique atmospheric altitudes are required."
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

        result = np.zeros_like(
            n_squared,
            dtype=float
        )

        result[:] = np.sqrt(
            n_squared
        )

        return result

    def atmospheric_profile(
        self,
        latitude: float,
        longitude: float,
        time_index: int
    ):
        data = self.load()

        if time_index < 0:
            raise IndexError(
                "Invalid meteorological time index."
            )

        if time_index >= len(
            data["time"]
        ):
            raise IndexError(
                "Invalid meteorological time index."
            )

        grid = self.nearest_grid(
            latitude,
            longitude
        )

        latitude_index = grid[
            "latitude_index"
        ]

        longitude_index = grid[
            "longitude_index"
        ]

        pressure_levels = data[
            "pressure_level"
        ]

        pressures = self.pressure_to_pa(
            pressure_levels
        )

        temperatures = data[
            "temperature"
        ][
            time_index,
            :,
            latitude_index,
            longitude_index
        ]

        geopotential = data[
            "geopotential"
        ][
            time_index,
            :,
            latitude_index,
            longitude_index
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

        order = np.argsort(
            altitude
        )

        brunt_by_original_index = np.empty_like(
            brunt_vaisala
        )

        brunt_by_original_index[
            order
        ] = brunt_vaisala

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
                        latitude_index,
                        longitude_index
                    ]
                ),
                "v": float(
                    data["v"][
                        time_index,
                        index,
                        latitude_index,
                        longitude_index
                    ]
                ),
                "temperature": float(
                    temperatures[index]
                ),
                "potential_temperature": float(
                    potential_temperatures[index]
                ),
                "brunt_vaisala": float(
                    brunt_by_original_index[index]
                )
            })

        return {
            "latitude": grid["latitude"],
            "longitude": grid["longitude"],
            "latitude_index": latitude_index,
            "longitude_index": longitude_index,
            "time_index": int(
                time_index
            ),
            "time": data["time"][
                time_index
            ],
            "profile": profile
        }

    def wind_at_altitude(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        time_index: int
    ):
        profile = self.atmospheric_profile(
            latitude=latitude,
            longitude=longitude,
            time_index=time_index
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
            )
        }

    def wind_profile(
        self,
        latitude: float,
        longitude: float,
        time_index: int
    ):
        profile = self.atmospheric_profile(
            latitude=latitude,
            longitude=longitude,
            time_index=time_index
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
                    "u": item[
                        "u"
                    ],
                    "v": item[
                        "v"
                    ]
                }
                for item in profile[
                    "profile"
                ]
            ]
        }

    def time_information(self):
        data = self.load()

        times = np.asarray(
            data["time"]
        )

        if len(times) == 0:
            raise ValueError(
                "Meteorological time data is empty."
            )

        times = times.astype(
            "datetime64[s]"
        )

        elapsed_seconds = (
            times - times[0]
        ).astype(
            "timedelta64[s]"
        ).astype(
            float
        )

        if len(times) == 1:
            return {
                "count": 1,
                "interval_seconds": None,
                "interval_hours": None,
                "intervals_seconds": np.asarray(
                    [],
                    dtype=float
                ),
                "elapsed_seconds": elapsed_seconds,
                "times": times
            }

        intervals = np.diff(
            elapsed_seconds
        )

        if np.any(
            intervals <= 0
        ):
            raise ValueError(
                "Meteorological times must be strictly increasing."
            )

        return {
            "count": len(times),
            "interval_seconds": float(
                intervals[0]
            ),
            "interval_hours": float(
                intervals[0] / 3600.0
            ),
            "intervals_seconds": intervals,
            "elapsed_seconds": elapsed_seconds,
            "times": times
        }

    def time_index_at_elapsed(
        self,
        elapsed_seconds: float
    ):
        information = (
            self.time_information()
        )

        elapsed = information[
            "elapsed_seconds"
        ]

        if elapsed_seconds < 0:
            raise ValueError(
                "Elapsed simulation time cannot be negative."
            )

        index = int(
            np.searchsorted(
                elapsed,
                elapsed_seconds,
                side="right"
            ) - 1
        )

        index = max(
            0,
            min(
                index,
                len(elapsed) - 1
            )
        )

        return index