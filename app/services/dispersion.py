import math


class DispersionService:
    def __init__(
        self,
        particle_simulation,
        meteorology,
        deposition=None,
        turbulence=None,
        surface_meteorology=None,
        landcover=None
    ):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology
        self.deposition = deposition
        self.turbulence = turbulence
        self.surface_meteorology = surface_meteorology
        self.landcover = landcover

    def _surface_conditions(self, lat, lon, alt, time_index):
        result = {}

        for name in (
            "conditions_at_position",
            "weather_at_position",
            "sample_at_position"
        ):
            fn = getattr(self.meteorology, name, None)

            if callable(fn):
                try:
                    value = fn(lat, lon, alt, time_index)

                    if isinstance(value, dict):
                        result.update(value)
                        break

                except (TypeError, ValueError, KeyError, IndexError):
                    pass

        if self.surface_meteorology:
            elapsed = self.meteorology.time_information()[
                "elapsed_seconds"
            ][time_index]

            surface_index = (
                self.surface_meteorology.time_index_at_elapsed(elapsed)
            )

            surface = self.surface_meteorology.surface_profile(
                lat, lon, surface_index
            )

            if isinstance(surface, dict):
                result.update(surface)

        if not result:
            try:
                value = self.meteorology.wind_at_altitude(
                    lat, lon, alt, time_index
                )

                if isinstance(value, dict):
                    result.update(value)

            except Exception:
                pass

        if self.landcover:
            result.update(self.landcover.surface_at(lat, lon))

        return result

    def simulate(
        self,
        particles,
        duration,
        dt=None,
        start_time_index=0
    ):
        duration = float(duration)

        if dt is None:
            dt = (
                self.turbulence.parameters()["timestep"]
                if self.turbulence else 71.25
            )

        dt = float(dt)

        if not particles or duration <= 0 or dt <= 0:
            raise ValueError("Particles, duration and dt must be valid.")

        info = self.meteorology.time_information()
        elapsed_times = info.get("elapsed_seconds", [])

        if not elapsed_times:
            raise ValueError("Meteorological time information is empty.")

        start_time_index = int(start_time_index)

        if not 0 <= start_time_index < len(elapsed_times):
            raise ValueError(
                "start_time_index exceeds meteorological dataset."
            )

        rng = getattr(self.particle_simulation, "rng", None)

        steps = math.ceil(duration / dt)
        trajectories = []

        for p in particles:
            lat = float(p["latitude"])
            lon = float(p["longitude"])
            alt = float(p["altitude"])

            trajectory = [{
                "time": 0.0,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "u": 0.0,
                "v": 0.0,
                "vertical": 0.0,
                "time_index": start_time_index
            }]

            status = "active"
            deposition = None
            fluct = [0.0, 0.0, 0.0]
            elapsed = 0.0

            while elapsed < duration - 1e-9:
                step_dt = min(dt, duration - elapsed)

                target = (
                    elapsed_times[start_time_index]
                    + elapsed + step_dt
                )

                ti = self.meteorology.time_index_at_elapsed(target)

                if self.turbulence:
                    params = self.turbulence.parameters()

                    fluct = [
                        self.turbulence.langevin_velocity(
                            fluct[i],
                            params["sigma_u"],
                            params["tau_l"],
                            step_dt,
                            rng
                        )
                        for i in range(3)
                    ]

                state = self.particle_simulation.step(
                    lat,
                    lon,
                    alt,
                    step_dt,
                    ti,
                    p.get("settling_velocity", 0.0),
                    fluct
                )

                lat = state["latitude"]
                lon = state["longitude"]
                alt = state["altitude"]
                elapsed = min(duration, elapsed + step_dt)

                trajectory.append({
                    "time": elapsed,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "u": state["u"],
                    "v": state["v"],
                    "vertical": state["vertical"],
                    "time_index": ti
                })

                if (
                    self.deposition
                    and alt <= 0.0
                    and str(
                        p.get("particle_class", p.get("class", ""))
                    ).lower() == "coarse"
                ):
                    status = "deposited"

                    deposition = {
                        "latitude": lat,
                        "longitude": lon,
                        "altitude": 0.0,
                        "time": elapsed,
                        "mass": float(p.get("mass", 0.0)),
                        "mechanism": "gravity",
                        "probability": 1.0,
                        "dry_probability": 0.0,
                        "wet_probability": 0.0,
                        "total_probability": 1.0,
                        "vd": float(p.get("settling_velocity", 0.0))
                    }

                    trajectory[-1]["altitude"] = 0.0
                    break

                if (
                    self.deposition
                    and alt <= getattr(
                        self.deposition, "layer_height", 100.0
                    )
                ):
                    cond = self._surface_conditions(
                        lat, lon, alt, ti
                    )

                    if self.landcover and "roughness_length" not in cond:
                        raise ValueError(
                            f"Roughness length ESA WorldCover tidak tersedia "
                            f"pada koordinat {lat}, {lon}."
                        )

                    friction_velocity = cond.get("friction_velocity")

                    if friction_velocity is None:
                        raise ValueError(
                            f"Friction velocity GFS tidak tersedia "
                            f"pada koordinat {lat}, {lon}."
                        )

                    friction_velocity = float(friction_velocity)

                    if (
                        not math.isfinite(friction_velocity)
                        or friction_velocity <= 0
                    ):
                        raise ValueError(
                            f"Friction velocity GFS tidak valid "
                            f"pada koordinat {lat}, {lon}."
                        )

                    rain = cond.get(
                        "precipitation_rate_mm_hour",
                        cond.get(
                            "rain_rate",
                            cond.get(
                                "precipitation",
                                cond.get("precip")
                            )
                        )
                    )

                    if rain is not None:
                        rain = float(rain)

                        if not math.isfinite(rain) or rain < 0:
                            raise ValueError(
                                "Curah hujan harus berupa nilai mm/jam valid."
                            )

                    rh = cond.get(
                        "relative_humidity",
                        cond.get("rh", cond.get("r"))
                    )

                    if rain is not None and rain > 0:
                        if rh is None:
                            raise ValueError(
                                f"Relative humidity GFS tidak tersedia pada koordinat {lat}, {lon}."
                            )

                        rh = float(rh)

                        if not math.isfinite(rh) or not 0 <= rh <= 100:
                            raise ValueError(
                                f"Relative humidity tidak valid: {rh}% pada koordinat {lat}, {lon}."
                            )
                    elif rh is not None:
                        rh = float(rh)

                    roughness = (
                        cond.get("roughness_length")
                        if self.landcover else None
                    )

                    dep = self.deposition.decide(
                        p,
                        alt,
                        step_dt,
                        friction_velocity=friction_velocity,
                        roughness_length=roughness,
                        rain_rate=rain,
                        rh=rh,
                        rng=rng
                    )

                    if dep["deposited"]:
                        status = "deposited"

                        deposition = {
                            "latitude": lat,
                            "longitude": lon,
                            "altitude": alt,
                            "time": elapsed,
                            "mass": float(p.get("mass", 0.0)),
                            "mechanism": dep["mechanism"],
                            "probability": dep["probability"],
                            "dry_probability": dep.get("pdry", 0.0),
                            "wet_probability": dep.get("pwet", 0.0),
                            "total_probability": dep.get("probability", 0.0),
                            "vd": dep.get("vd", 0.0),
                            "ra": dep.get("ra"),
                            "rs": dep.get("rs"),
                            "relative_humidity": rh,
                            "rain_rate_mm_hour": rain,
                            "scavenging_type": dep.get("scavenging_type"),
                            "scavenging_coefficient": dep.get("scavenging_coefficient"),
                            "pdry": dep.get("pdry", 0.0),
                            "pwet": dep.get("pwet", 0.0),
                            "friction_velocity": dep.get(
                                "friction_velocity"
                            ),
                            "roughness_length": dep.get(
                                "roughness_length"
                            ),
                            "landcover_class": cond.get(
                                "landcover_class"
                            ),
                            "landcover_name": cond.get(
                                "landcover_name"
                            ),
                            "landcover_tile": cond.get(
                                "landcover_tile"
                            )
                        }

                        break

            trajectories.append({
                **p,
                "trajectory": trajectory,
                "status": status,
                "deposition": deposition
            })

        return {
            "particle_count": len(trajectories),
            "total_particles": len(trajectories),
            "duration": duration,
            "dt": dt,
            "steps": steps,
            "start_time_index": start_time_index,
            "trajectories": trajectories,
            "meteorology": info
        }