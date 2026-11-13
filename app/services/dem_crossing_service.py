
import math
import numpy as np


class DEMCrossingService:
    def __init__(
        self,
        particle_simulation,
        terrain,
        spatial_resolution=10.0,
        min_substeps=8,
        max_substeps=256,
        time_tolerance=0.001
    ):
        if terrain is None:
            raise ValueError("TerrainService wajib tersedia.")

        if spatial_resolution <= 0:
            raise ValueError("spatial_resolution harus positif.")

        if min_substeps < 1:
            raise ValueError("min_substeps minimal 1.")

        if max_substeps < min_substeps:
            raise ValueError(
                "max_substeps harus >= min_substeps."
            )

        if time_tolerance <= 0:
            raise ValueError("time_tolerance harus positif.")

        self.simulation = particle_simulation
        self.terrain = terrain
        self.spatial_resolution = float(spatial_resolution)
        self.min_substeps = int(min_substeps)
        self.max_substeps = int(max_substeps)
        self.time_tolerance = float(time_tolerance)

        self.stats = {
            "detect_calls": 0,
            "position_calls": 0,
            "dem_calls": 0,
            "crossings": 0,
            "samples": 0
        }

    def _elevation(self, lat, lon):
        self.stats["dem_calls"] += 1

        elevation = float(
            self.terrain.elevation_at(lat, lon)["elevation"]
        )

        if not math.isfinite(elevation):
            raise ValueError("Elevasi DEM tidak valid.")

        return elevation

    def _position(
        self,
        lat,
        lon,
        alt,
        dt,
        elapsed,
        settling_velocity,
        fluct
    ):
        self.stats["position_calls"] += 1

        if dt == 0:
            wind = self.simulation.velocity(
                lat,
                lon,
                alt,
                float(elapsed)
            )

            return {
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude": float(alt),
                "u": wind["u"] + fluct[0],
                "v": wind["v"] + fluct[1],
                "vertical": (
                    wind["w"]
                    + fluct[2]
                    - settling_velocity
                )
            }

        return self.simulation.step(
            lat,
            lon,
            alt,
            dt,
            float(elapsed),
            settling_velocity,
            fluct
        )

    def _sample(
        self,
        lat,
        lon,
        alt,
        dt,
        elapsed,
        settling_velocity,
        fluct
    ):
        self.stats["samples"] += 1

        result = self._position(
            lat,
            lon,
            alt,
            dt,
            elapsed,
            settling_velocity,
            fluct
        )

        y = float(result["latitude"])
        x = float(result["longitude"])
        z = float(result["altitude"])

        ground = self._elevation(y, x)

        return {
            "latitude": y,
            "longitude": x,
            "altitude": z,
            "ground": ground,
            "agl": z - ground,
            "offset": float(dt),
            "u": float(result["u"]),
            "v": float(result["v"]),
            "vertical": float(result["vertical"])
        }

    @staticmethod
    def _distance(a, b):
        radius = 6371000.0

        lat1 = math.radians(a["latitude"])
        lat2 = math.radians(b["latitude"])

        dlat = lat2 - lat1
        dlon = math.radians(
            b["longitude"] - a["longitude"]
        )

        value = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        return 2 * radius * math.asin(
            min(1.0, math.sqrt(value))
        )

    def detect(
        self,
        lat,
        lon,
        alt,
        dt,
        elapsed,
        settling_velocity,
        fluct=None
    ):
        self.stats["detect_calls"] += 1

        if fluct is None:
            fluct = np.zeros(3)

        dt = float(dt)

        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt tidak valid.")

        start = self._sample(
            lat,
            lon,
            alt,
            0.0,
            elapsed,
            settling_velocity,
            fluct
        )

        if start["agl"] <= 0:
            self.stats["crossings"] += 1

            return {
                "crossing": True,
                "fraction": 0.0,
                "effective_dt": 0.0,
                "state": start
            }

        if dt == 0:
            return {
                "crossing": False,
                "fraction": 1.0,
                "effective_dt": 0.0,
                "state": start
            }

        end = self._sample(
            lat,
            lon,
            alt,
            dt,
            elapsed,
            settling_velocity,
            fluct
        )

        distance = self._distance(start, end)

        count = min(
            self.max_substeps,
            max(
                self.min_substeps,
                math.ceil(
                    distance / self.spatial_resolution
                )
            )
        )

        previous = start

        for i in range(1, count + 1):
            offset = dt * i / count

            current = (
                end
                if i == count
                else self._sample(
                    lat,
                    lon,
                    alt,
                    offset,
                    elapsed,
                    settling_velocity,
                    fluct
                )
            )

            if (
                previous["agl"] > 0
                and current["agl"] <= 0
            ):
                left = previous["offset"]
                right = current["offset"]

                while right - left > self.time_tolerance:
                    middle = (left + right) / 2

                    sample = self._sample(
                        lat,
                        lon,
                        alt,
                        middle,
                        elapsed,
                        settling_velocity,
                        fluct
                    )

                    if sample["agl"] <= 0:
                        right = middle
                    else:
                        left = middle

                crossing_state = self._sample(
                    lat,
                    lon,
                    alt,
                    right,
                    elapsed,
                    settling_velocity,
                    fluct
                )

                crossing_state["altitude"] = (
                    crossing_state["ground"]
                )
                crossing_state["agl"] = 0.0

                self.stats["crossings"] += 1

                return {
                    "crossing": True,
                    "fraction": right / dt,
                    "effective_dt": right,
                    "state": crossing_state
                }

            previous = current

        return {
            "crossing": False,
            "fraction": 1.0,
            "effective_dt": dt,
            "state": end
        }
