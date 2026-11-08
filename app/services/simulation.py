
from typing import Any
from app.services.landcover import LandCoverService

class SimulationService:
    def __init__(
        self, meteorology, eruption, particle, dispersion,
        geojson, concentration=None, landcover=None
    ):
        self.meteorology = meteorology
        self.eruption = eruption
        self.particle = particle
        self.dispersion = dispersion
        self.geojson = geojson
        self.concentration = concentration

        self.landcover = landcover
        self.dispersion.landcover = self.landcover

    def run(self, config) -> dict[str, Any]:
        info = self.meteorology.time_information()

        if not info.get("times"):
            raise ValueError("Meteorological dataset contains no valid time.")

        time_index = self._resolve_time_index(config, info)

        atmospheric = self.meteorology.atmospheric_profile(
            config.latitude, config.longitude, time_index
        )

        eruption = self.eruption.calculate(
            atmospheric, config.eruption_height, config.eruption_duration
        )

        summary = self.particle.create_particle_summary(eruption["total_mass"])

        particles = self.particle.create_particles(
            eruption["total_mass"],
            config.latitude,
            config.longitude,
            config.altitude
        )

        grid = self.meteorology.nearest_grid(
            config.latitude, config.longitude
        )

        auto_timestep = self._get(config, "auto_timestep", True)

        if auto_timestep and getattr(self.dispersion, "turbulence", None) is not None:
            effective_dt = float(
                self.dispersion.turbulence.parameters()["timestep"]
            )
        else:
            effective_dt = float(self._get(config, "timestep", 71.25))

        disp = self.dispersion.simulate(
            particles,
            self._get(config, "duration"),
            effective_dt,
            time_index
        )

        trajectories = disp["trajectories"]

        deposited = [
            p for p in trajectories if p.get("status") == "deposited"
        ]
        active = [
            p for p in trajectories if p.get("status") == "active"
        ]

        source_mass = float(eruption["total_mass"])
        deposited_mass = sum(float(p.get("mass", 0.0)) for p in deposited)
        airborne_mass = sum(float(p.get("mass", 0.0)) for p in active)

        mass_error = source_mass - deposited_mass - airborne_mass

        mechanism_mass = {}
        scavenging_mass = {}
        scavenging_particles = {}

        for p in deposited:
            dep = p.get("deposition") or {}
            mass = float(p.get("mass", 0.0))

            mechanism = dep.get("mechanism") or "unknown"
            mechanism_mass[mechanism] = mechanism_mass.get(mechanism, 0.0) + mass

            kind = dep.get("scavenging_type")
            if kind:
                scavenging_mass[kind] = scavenging_mass.get(kind, 0.0) + mass
                scavenging_particles[kind] = scavenging_particles.get(kind, 0) + 1

        for p in deposited:
            dep = p.get("deposition") or {}
            kind = dep.get("scavenging_type")

            if kind:
                scavenging_mass[kind] = (
                    scavenging_mass.get(kind, 0.0) + float(p.get("mass", 0.0))
                )
                scavenging_particles[kind] = (
                    scavenging_particles.get(kind, 0) + 1
                )

        grid_size = 0.25
        vertical_resolution = 500.0

        concentration = (
            self.concentration.accumulate(
                trajectories,
                config.latitude,
                config.longitude,
                dlat=grid_size,
                dlon=grid_size,
                dz=vertical_resolution,
                active_only=True
            )
            if self.concentration else None
        )

        surface = (
            self.concentration.surface_deposition(
                deposited,
                dlat=grid_size,
                dlon=grid_size,
                origin_latitude=config.latitude,
                origin_longitude=config.longitude
            )
            if self.concentration else None
        )

        spatial = (
            self.concentration.plume_metrics(
                eruption["mean_wind_speed"],
                self._get(config, "duration"),
                dlat=grid_size,
                dlon=grid_size,
                latitude=config.latitude
            )
            if self.concentration else None
        )

        trajectory_geojson = self.geojson.trajectory_to_feature_collection(disp)
        point_geojson = self.geojson.points_to_feature_collection(disp)
        deposition_geojson = self.geojson.deposition_to_feature_collection(disp)

        grid_geojson = (
            self.geojson.surface_deposition_to_feature_collection(surface)
            if surface is not None
            else {"type": "FeatureCollection", "features": []}
        )

        closure = (
            100.0 * (deposited_mass + airborne_mass) / source_mass
            if source_mass > 0 else None
        )

        serialized_config = self._serialize_config(config)
        serialized_config["grid_size"] = grid_size
        serialized_config["vertical_resolution"] = vertical_resolution
        serialized_config["timestep"] = effective_dt
        serialized_config["auto_timestep"] = bool(auto_timestep)

        return {
            "config": serialized_config,
            "meteorology": {
                **info,
                "time_index": time_index,
                "times": [str(t) for t in info["times"]]
            },
            "location": {
                "latitude": config.latitude,
                "longitude": config.longitude,
                "altitude": config.altitude,
                "grid": grid
            },
            "atmospheric": atmospheric,
            "eruption": eruption,
            "particle_summary": summary,
            "particles": {
                "total": len(particles),
                "groups": particles
            },
            "dispersion": {
                "particle_count": len(trajectories),
                "duration": config.duration,
                "dt": effective_dt,
                "steps": disp["steps"],
                "trajectories": trajectories
            },
            "concentration": concentration,
            "surface_deposition": surface,
            "spatial_analysis": spatial,
            "mass_balance": {
                "source_mass_kg": source_mass,
                "deposited_mass_kg": deposited_mass,
                "airborne_mass_kg": airborne_mass,
                "residual_kg": mass_error,
                "closure_percent": closure,
                "deposition_mechanism_mass_kg": mechanism_mass,
                "scavenging_mass_kg": scavenging_mass,
                "scavenging_particles": scavenging_particles,
                "total": len(trajectories),
                "deposited": len(deposited),
                "active": len(active),
                "total_particles": len(trajectories),
                "deposited_particles": len(deposited),
                "active_particles": len(active)
            },
            "geojson": {
                "trajectory": trajectory_geojson,
                "points": point_geojson,
                "deposition": deposition_geojson,
                "deposition_grid": grid_geojson
            }
        }

    @staticmethod
    def _resolve_time_index(config, info):
        index = getattr(config, "time_index", 0)

        if index is None:
            index = 0

        if not 0 <= int(index) < int(
            info.get("count", len(info.get("times", [])))
        ):
            raise ValueError("time_index exceeds meteorological dataset.")

        return int(index)

    @staticmethod
    def _get(config, key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def _serialize_config(config):
        return (
            config.model_dump()
            if hasattr(config, "model_dump")
            else dict(config)
        )

    def close(self):
        if self.landcover:
            self.landcover.close()
