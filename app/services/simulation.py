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

    def run(
        self,
        config
    ) -> dict[str, Any]:

        # --------------------------------------------------
        # 1. Resolve simulation time
        # --------------------------------------------------

        time_information = (
            self.meteorology.time_information()
        )

        if not time_information["times"]:
            raise ValueError(
                "Meteorological dataset contains no valid time."
            )

        time_index = (
            self._resolve_time_index(
                config=config,
                time_information=time_information
            )
        )

        # --------------------------------------------------
        # 2. Atmospheric profile
        # --------------------------------------------------

        atmospheric = (
            self.meteorology.atmospheric_profile(
                latitude=config.latitude,
                longitude=config.longitude,
                time_index=time_index
            )
        )

        # --------------------------------------------------
        # 3. Eruption source calculation
        # --------------------------------------------------

        eruption = (
            self.eruption.calculate(
                atmospheric_profile=atmospheric,
                eruption_height=config.eruption_height,
                eruption_duration=config.eruption_duration
            )
        )

        # --------------------------------------------------
        # 4. Particle distribution
        # --------------------------------------------------

        particle_summary = (
            self.particle.create_particle_summary(
                total_mass=eruption["total_mass"]
            )
        )

        particles = (
            self.particle.create_particles(
                total_mass=eruption["total_mass"],
                latitude=config.latitude,
                longitude=config.longitude,
                altitude=config.altitude
            )
        )

        # --------------------------------------------------
        # 5. Meteorological grid
        # --------------------------------------------------

        grid = (
            self.meteorology.nearest_grid(
                latitude=config.latitude,
                longitude=config.longitude
            )
        )

        # --------------------------------------------------
        # 6. Particle dispersion
        # --------------------------------------------------

        dispersion = (
            self.dispersion.simulate(
                particles=particles,
                duration=config.duration,
                dt=config.timestep
            )
        )

        # --------------------------------------------------
        # 7. GeoJSON
        # --------------------------------------------------

        trajectory_geojson = (
            self.geojson
            .trajectory_to_feature_collection(
                dispersion
            )
        )

        point_geojson = (
            self.geojson
            .trajectory_points_to_feature_collection(
                dispersion
            )
        )

        # --------------------------------------------------
        # 8. Particle count
        # --------------------------------------------------

        total_particles = sum(
            particle.get("count", 1)
            for particle in particles
        )

        # --------------------------------------------------
        # 9. Build result
        # --------------------------------------------------

        return {
            "config": self._serialize_config(
                config
            ),

            "meteorology": {
                "time_index": time_index,
                "time_count": time_information[
                    "count"
                ],
                "interval_seconds": (
                    time_information[
                        "interval_seconds"
                    ]
                ),
                "interval_hours": (
                    time_information[
                        "interval_hours"
                    ]
                ),
                "times": [
                    str(value)
                    for value in time_information[
                        "times"
                    ]
                ]
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
                "total": total_particles,
                "groups": particles
            },

            "dispersion": {
                "particle_count": dispersion[
                    "particle_count"
                ],
                "duration": dispersion[
                    "duration"
                ],
                "dt": dispersion[
                    "dt"
                ],
                "steps": dispersion[
                    "steps"
                ],
                "trajectories": dispersion[
                    "trajectories"
                ]
            },

            "geojson": {
                "trajectory": trajectory_geojson,
                "points": point_geojson
            }
        }

    def _resolve_time_index(
        self,
        config,
        time_information
    ) -> int:

        requested = getattr(
            config,
            "time_index",
            None
        )

        if requested is not None:

            requested = int(
                requested
            )

            if requested < 0:
                raise ValueError(
                    "time_index cannot be negative."
                )

            if requested >= time_information[
                "count"
            ]:
                raise ValueError(
                    "time_index exceeds meteorological dataset."
                )

            return requested

        return 0

    @staticmethod
    def _serialize_config(
        config
    ) -> dict[str, Any]:

        if hasattr(
            config,
            "model_dump"
        ):
            return config.model_dump()

        return dict(
            config
        )