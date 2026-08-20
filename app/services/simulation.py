from typing import Any


class SimulationService:

    def __init__(
        self,
        meteorology,
        eruption,
        particle,
        dispersion,
        geojson
    ):
        self.meteorology = meteorology
        self.eruption = eruption
        self.particle = particle
        self.dispersion = dispersion
        self.geojson = geojson

    def run(self, config) -> dict[str, Any]:

        info = self.meteorology.time_information()

        if not info["times"]:
            raise ValueError("No meteorological time available.")

        time_index = self._resolve_time_index(config, info)

        atmospheric = self.meteorology.atmospheric_profile(
            config.latitude,
            config.longitude,
            time_index
        )

        eruption = self.eruption.calculate(
            atmospheric,
            config.eruption_height,
            config.eruption_duration
        )

        particle_summary = self.particle.create_particle_summary(
            eruption["total_mass"]
        )

        particles = self.particle.create_particles(
            eruption["total_mass"],
            config.latitude,
            config.longitude,
            config.altitude
        )

        grid = self.meteorology.nearest_grid(
            config.latitude,
            config.longitude
        )

        dispersion = self.dispersion.simulate(
            particles=particles,
            duration=config.duration,
            dt=config.timestep,
            start_time_index=time_index
        )

        trajectory_geojson = (
            self.geojson
            .trajectory_to_feature_collection(dispersion)
        )

        point_geojson = (
            self.geojson
            .trajectory_points_to_feature_collection(dispersion)
        )

        statistics = self._statistics(
            dispersion,
            config.latitude,
            config.longitude
        )

        return {
            "config": self._serialize_config(config),

            "meteorology": {
                "time_index": time_index,
                "time_count": info["count"],
                "interval_seconds":
                    info["interval_seconds"],
                "interval_hours":
                    info["interval_hours"],
                "times": [str(x) for x in info["times"]]
            },

            "location": {
                "latitude": config.latitude,
                "longitude": config.longitude,
                "altitude": config.altitude,
                "grid": grid
            },

            "atmospheric": atmospheric,
            "eruption": eruption,

            "particle_summary": particle_summary,

            "particles": {
                "total": len(particles),
                "groups": particle_summary["groups"]
            },

            "dispersion": dispersion,

            "mass_balance": {
                "initial_mass":
                    dispersion["total_mass"],
                "deposited_mass":
                    dispersion["deposited_mass"],
                "airborne_mass":
                    dispersion["airborne_mass"],
                "error":
                    dispersion["mass_error"],
                "error_percent":
                    (
                        abs(dispersion["mass_error"])
                        / max(dispersion["total_mass"], 1e-30)
                        * 100.0
                    )
            },

            "statistics": statistics,

            "geojson": {
                "trajectory": trajectory_geojson,
                "points": point_geojson
            }
        }

    def _statistics(self, dispersion, source_lat, source_lon):

        max_distance = 0.0
        max_altitude = 0.0
        deposited = []

        for p in dispersion["trajectories"]:
            for x in p["trajectory"]:

                max_altitude = max(
                    max_altitude,
                    float(x["altitude"])
                )

                distance = self._distance(
                    source_lat,
                    source_lon,
                    x["latitude"],
                    x["longitude"]
                )

                max_distance = max(
                    max_distance,
                    distance
                )

            if p.get("deposition"):
                deposited.append(p["deposition"])

        return {
            "max_distance_m":
                max_distance,
            "max_distance_km":
                max_distance / 1000.0,
            "max_altitude_m":
                max_altitude,
            "deposited_particles":
                len(deposited)
        }

    @staticmethod
    def _distance(lat1, lon1, lat2, lon2):

        import math

        r = 6371000.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)

        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)

        a = (
            math.sin(dp / 2) ** 2
            + math.cos(p1)
            * math.cos(p2)
            * math.sin(dl / 2) ** 2
        )

        return 2 * r * math.asin(
            min(1.0, math.sqrt(a))
        )

    @staticmethod
    def _resolve_time_index(config, info):

        index = getattr(config, "time_index", 0)

        index = int(index)

        if not 0 <= index < info["count"]:
            raise ValueError(
                f"time_index must be between 0 and {info['count'] - 1}"
            )

        return index

    @staticmethod
    def _serialize_config(config):

        if hasattr(config, "model_dump"):
            return config.model_dump()

        return dict(config)