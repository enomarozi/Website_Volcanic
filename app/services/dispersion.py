
import math

class DispersionService:
    def __init__(self, particle_simulation, meteorology, deposition=None,
                 turbulence=None, surface_meteorology=None, landcover=None,
                 terrain=None):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology
        self.deposition = deposition
        self.turbulence = turbulence
        self.surface_meteorology = surface_meteorology
        self.landcover = landcover
        self.terrain = terrain

    def _terrain_elevation(self, lat, lon):
        if self.terrain is None:
            raise RuntimeError("TerrainService belum dikonfigurasi.")
        result = self.terrain.elevation_at(lat, lon)
        elevation = float(result["elevation"])
        if not math.isfinite(elevation):
            raise ValueError("Elevasi DEM tidak valid.")
        return elevation

    @staticmethod
    def _number(value, name, minimum=None, maximum=None):
        if value is None:
            raise ValueError(f"{name} tidak tersedia.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} tidak valid.")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} lebih kecil dari {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} lebih besar dari {maximum}.")
        return value

    @staticmethod
    def _first(data, *keys):
        for key in keys:
            if data.get(key) is not None:
                return data[key]
        return None

    def _surface_conditions(self, lat, lon, alt, time_index,
                        elapsed_seconds, include_landcover=False):
        result = {}
        for name in ("conditions_at_position", "weather_at_position",
                     "sample_at_position"):
            fn = getattr(self.meteorology, name, None)
            if callable(fn):
                try:
                    value = fn(lat, lon, alt, time_index)
                except (TypeError):
                    continue
                if isinstance(value, dict):
                    result.update(value)
                    break

        if self.surface_meteorology:
            surface_index = self.surface_meteorology.time_index_at_elapsed(
                elapsed_seconds
            )

            rain_index = self.surface_meteorology.precipitation_index_at_elapsed(
                elapsed_seconds
            )

            surface = self.surface_meteorology.surface_profile(
                lat,
                lon,
                surface_index,
                precipitation_index=rain_index
            )

            if isinstance(surface, dict):
                result.update(surface)

        if include_landcover and self.landcover:
            result.update(self.landcover.surface_at(lat, lon))
        return result

    def _rain_and_humidity(self, conditions):
        rain = self._first(
            conditions, "precipitation_rate_mm_hour", "rain_rate"
        )
        if rain is None:
            raise ValueError(
                "Laju hujan mm/jam tidak tersedia. "
                "Data akumulasi presipitasi harus dikonversi menjadi laju."
            )
        rain = self._number(rain, "Laju hujan mm/jam", minimum=0.0)
        rh = self._first(conditions, "relative_humidity", "rh")
        if rain > 0:
            rh = self._number(rh, "Relative humidity", 0.0, 100.0)
        elif rh is not None:
            rh = self._number(rh, "Relative humidity", 0.0, 100.0)
        return rain, rh

    def _record(self, p, lat, lon, alt, elapsed, mechanism,
                probability=1.0, pdry=0.0, pwet=0.0, **extra):
        return {
            "latitude": float(lat),
            "longitude": float(lon),
            "altitude": float(alt),
            "time": float(elapsed),
            "mass": float(p.get("mass", 0.0)),
            "mechanism": mechanism,
            "probability": float(probability),
            "dry_probability": float(pdry),
            "wet_probability": float(pwet),
            "total_probability": float(probability),
            "pdry": float(pdry),
            "pwet": float(pwet),
            "vd": float(extra.pop("vd", p.get("settling_velocity", 0.0))),
            **extra
        }

    def simulate(self, particles, duration, dt=None, start_time_index=0):
        if self.terrain is None:
            raise RuntimeError("TerrainService wajib tersedia.")
        if not particles:
            raise ValueError("Particles tidak boleh kosong.")

        params = self.turbulence.parameters() if self.turbulence else None
        duration = self._number(duration, "Duration", minimum=0.0)
        dt = self._number(
            params["timestep"] if dt is None and params else
            71.25 if dt is None else dt,
            "Timestep", minimum=0.0
        )
        if duration == 0 or dt == 0:
            raise ValueError("Duration dan timestep harus positif.")

        info = self.meteorology.time_information()
        times = info.get("elapsed_seconds", [])
        start_time_index = int(start_time_index)
        if not times or not 0 <= start_time_index < len(times):
            raise ValueError("Time index meteorologi tidak valid.")

        rng = getattr(self.particle_simulation, "rng", None)
        if rng is None:
            raise RuntimeError("Random generator partikel tidak tersedia.")

        steps = math.ceil(duration / dt)
        trajectories = []
        start_time = float(times[start_time_index])
        if start_time+duration>float(times[-1])+1e-6:
            raise ValueError(
                "Durasi simulasi melewati waktu terakhir data meteorologi GFS."
            )
        for p in particles:
            lat = float(p["latitude"])
            lon = float(p["longitude"])
            alt = float(p["altitude"])
            ground = self._terrain_elevation(lat, lon)
            if alt < ground:
                raise ValueError(
                    f"Partikel dimulai di bawah DEM: "
                    f"altitude={alt:.2f} m, ground={ground:.2f} m, "
                    f"koordinat=({lat}, {lon})."
                )

            release_time = self._number(
                p.get("release_time", 0.0),
                "Release time",
                minimum=0.0
            )

            if release_time >= duration:
                raise ValueError(
                    "Release time harus lebih kecil dari durasi simulasi."
                )

            release_index = self.meteorology.time_index_at_elapsed(
                start_time + release_time
            )

            trajectory = [{
                "time": release_time,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "u": 0.0,
                "v": 0.0,
                "vertical": 0.0,
                "time_index": release_index
            }]

            status = "active"
            deposition = None
            fluct = [0.0, 0.0, 0.0]
            elapsed = release_time

            while elapsed < duration - 1e-9:
                next_boundary = min(
                    duration,
                    (math.floor((elapsed + 1e-9) / dt) + 1) * dt
                )

                step_dt = min(
                    duration - elapsed,
                    next_boundary - elapsed
                )

                if step_dt <= 1e-9:
                    step_dt = min(dt, duration - elapsed)
                target=start_time+elapsed+step_dt
                ti=self.meteorology.time_index_at_elapsed(target)
                wind_time=float(start_time+elapsed)

                if params:
                    mixing_height = float(params["mixing_height"])
                    height_agl = max(0.0, alt - ground)
                    sigma = float(params["sigma_u"])
                    tau = float(params["tau_l"])

                    if height_agl > mixing_height:
                        sigma *= math.exp(
                            -(height_agl - mixing_height) / mixing_height
                        )

                    fluct = [
                        self.turbulence.langevin_velocity(
                            fluct[i], sigma, tau, step_dt, rng
                        )
                        for i in range(3)
                    ]

                old_lat, old_lon, old_alt = lat, lon, alt
                old_agl = old_alt - ground

                state=self.particle_simulation.step(
                    lat,lon,alt,step_dt,wind_time,
                    p.get("settling_velocity",0.0),fluct
                )
                new_lat = float(state["latitude"])
                new_lon = float(state["longitude"])
                new_alt = float(state["altitude"])
                new_ground = self._terrain_elevation(new_lat, new_lon)
                new_agl = new_alt - new_ground

                crossing = new_agl <= 0.0
                fraction = 1.0
                if crossing:
                    denominator = old_agl - new_agl
                    fraction = (
                        max(0.0, min(1.0, old_agl / denominator))
                        if denominator > 0 else 0.0
                    )

                effective_dt = step_dt * fraction
                end_time = elapsed + effective_dt

                if crossing:
                    lat = old_lat + fraction * (new_lat - old_lat)
                    delta_lon = ((new_lon - old_lon + 180.0) % 360.0) - 180.0
                    lon = ((old_lon + fraction * delta_lon + 180.0) % 360.0) - 180.0
                    ground = self._terrain_elevation(lat, lon)
                    alt = ground
                else:
                    lat, lon, alt, ground = (
                        new_lat, new_lon, new_alt, new_ground
                    )

                height_agl = max(0.0, alt - ground)
                conditions = {}
                dep = None

                if self.deposition and not crossing and effective_dt > 0:
                    near_surface = height_agl <= self.deposition.layer_height

                    conditions = self._surface_conditions(
                        lat,
                        lon,
                        alt,
                        ti,
                        elapsed_seconds=start_time + elapsed + effective_dt / 2.0,
                        include_landcover=near_surface
                    )

                    rain, rh = self._rain_and_humidity(conditions)

                    friction = None
                    roughness = None

                    if near_surface:
                        friction = self._number(
                            conditions.get("friction_velocity"),
                            "Friction velocity",
                            minimum=0.0
                        )

                        if friction <= 0:
                            raise ValueError(
                                "Friction velocity harus positif."
                            )

                        roughness = conditions.get("roughness_length")

                        if self.landcover and roughness is None:
                            raise ValueError(
                                f"Roughness ESA tidak tersedia: {lat}, {lon}"
                            )

                    dep = self.deposition.decide(
                        particle=p,
                        altitude=height_agl,
                        dt=effective_dt,
                        friction_velocity=friction,
                        roughness_length=roughness,
                        rain_rate=rain,
                        rh=rh,
                        rng=rng
                    )

                if dep and dep["deposited"]:
                    status = "deposited"
                    deposition = self._record(
                        p, lat, lon, alt, end_time, dep["mechanism"],
                        probability=dep["probability"],
                        pdry=dep.get("pdry", 0.0),
                        pwet=dep.get("pwet", 0.0),
                        vd=dep.get("vd", 0.0),
                        ra=dep.get("ra"),
                        rs=dep.get("rs"),
                        relative_humidity=rh,
                        rain_rate_mm_hour=rain,
                        scavenging_type=dep.get("scavenging_type"),
                        scavenging_coefficient=dep.get("scavenging_coefficient"),
                        friction_velocity=dep.get("friction_velocity"),
                        roughness_length=dep.get("roughness_length"),
                        landcover_class=conditions.get("landcover_class"),
                        landcover_name=conditions.get("landcover_name"),
                        landcover_tile=conditions.get("landcover_tile")
                    )

                if status == "active" and crossing:
                    status = "deposited"
                    deposition = self._record(
                        p, lat, lon, ground, end_time,
                        "gravitational",
                        vd=p.get("settling_velocity", 0.0)
                    )

                elapsed = end_time if crossing else min(
                    duration, elapsed + step_dt
                )
                trajectory.append({
                    "time": elapsed,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "u": state["u"],
                    "v": state["v"],
                    "vertical": state["vertical"],
                    "time_index": self.meteorology.time_index_at_elapsed(
                        start_time + elapsed
                    )
                })

                if status == "deposited":
                    break

            trajectories.append({
                **p, "trajectory": trajectory,
                "status": status, "deposition": deposition
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
