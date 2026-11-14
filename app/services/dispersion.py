
import math
import time
from app.services.dem_crossing_service import DEMCrossingService


class DispersionService:
    def __init__(
        self,
        particle_simulation,
        meteorology,
        deposition=None,
        turbulence=None,
        surface_meteorology=None,
        landcover=None,
        terrain=None
    ):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology
        self.deposition = deposition
        self.turbulence = turbulence
        self.surface_meteorology = surface_meteorology
        self.landcover = landcover
        self.terrain = terrain
        self.dem_crossing = (
            DEMCrossingService(
                particle_simulation=self.particle_simulation,
                terrain=self.terrain,
                spatial_resolution=10.0,
                min_substeps=8,
                max_substeps=256,
                time_tolerance=0.001
            )
            if self.terrain is not None
            else None
        )
        self._perf = None

    def _terrain_elevation(self, lat, lon):
        if self.terrain is None:
            raise RuntimeError("TerrainService belum dikonfigurasi.")

        start = time.perf_counter()
        try:
            result = self.terrain.elevation_at(lat, lon)
        finally:
            if self._perf is not None:
                self._perf["terrain_time"] += time.perf_counter() - start
                self._perf["terrain_calls"] += 1

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

    def _surface_conditions(
        self,
        lat,
        lon,
        alt,
        time_index,
        elapsed_seconds,
        include_landcover=False
    ):
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
                except TypeError:
                    continue

                if isinstance(value, dict):
                    result.update(value)
                break

        if self.surface_meteorology:
            surface_index = (
                self.surface_meteorology.time_index_at_elapsed(
                    elapsed_seconds
                )
            )

            rain_index = (
                self.surface_meteorology.precipitation_index_at_elapsed(
                    elapsed_seconds
                )
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
            start = time.perf_counter()

            try:
                result.update(self.landcover.surface_at(lat, lon))
            finally:
                if self._perf is not None:
                    self._perf["landcover_time"] += (
                        time.perf_counter() - start
                    )
                    self._perf["landcover_calls"] += 1

        return result

    
    def _rain_and_humidity(self, conditions):
        source = getattr(
            self,
            "precipitation_source",
            "dataset"
        )

        if source == "none":
            rain = 0.0

        elif source == "manual":
            rain = self._number(
                getattr(self, "precipitation_rate", None),
                "Laju hujan manual mm/jam",
                minimum=0.0
            )

        elif source == "dataset":
            rain = self._first(
                conditions,
                "precipitation_rate_mm_hour",
                "rain_rate"
            )

            if rain is None:
                raise ValueError(
                    "Laju hujan GFS dalam mm/jam tidak tersedia."
                )

            rain = self._number(
                rain,
                "Laju hujan GFS mm/jam",
                minimum=0.0
            )

        else:
            raise ValueError(
                f"Precipitation source tidak dikenal: {source}"
            )

        rh = self._first(
            conditions,
            "relative_humidity",
            "rh"
        )

        if rh is not None:
            rh = self._number(
                rh,
                "Relative humidity",
                minimum=0.0,
                maximum=100.0
            )

        if rain > 0 and rh is None:
            raise ValueError(
                "Relative humidity diperlukan untuk "
                "perhitungan wet scavenging."
            )

        return rain, rh


    def _record(
        self,
        p,
        lat,
        lon,
        alt,
        elapsed,
        mechanism,
        probability=1.0,
        pdry=0.0,
        pwet=0.0,
        **extra
    ):
        return {
            "particle_id": p.get("particle_id", p.get("id")),
            "particle_class": p.get("particle_class", p.get("class")),
            "radius": float(p["radius"]),
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
            "vd": float(
                extra.pop("vd", p.get("settling_velocity", 0.0))
            ),
            **extra
        }

    def _print_performance(self, total_particles, steps):
        perf = self._perf

        if perf is None:
            return

        elapsed = time.perf_counter() - perf["start"]

        print("\n========== DISPERSION PERFORMANCE ==========")
        print(f"Particles           : {total_particles}")
        print(f"Transport steps     : {steps}")
        print(f"Transport iterations: {perf['transport_iterations']}")
        print(f"Deposition substeps : {perf['deposition_substeps']}")
        print(f"DEM crossing calls  : {perf['dem_crossing_calls']}")
        print(f"DEM crossing time   : {perf['dem_crossing_time']:.3f} s")
        print(f"DEM direct calls    : {perf['terrain_calls']}")
        print(f"DEM direct time     : {perf['terrain_time']:.3f} s")
        print(f"Surface calls       : {perf['surface_calls']}")
        print(f"Surface time        : {perf['surface_time']:.3f} s")
        print(f"Landcover calls     : {perf['landcover_calls']}")
        print(f"Landcover time      : {perf['landcover_time']:.3f} s")
        print(f"Particle step calls : {perf['particle_step_calls']}")
        print(f"Particle step time  : {perf['particle_step_time']:.3f} s")
        print(f"Event rate calls    : {perf['event_rate_calls']}")
        print(f"Event rate time     : {perf['event_rate_time']:.3f} s")
        print(f"Deposition loop time: {perf['deposition_loop_time']:.3f} s")
        print(f"Total elapsed       : {elapsed:.3f} s")
        if self.surface_meteorology:
            cache = self.surface_meteorology._surface_profile_cached.cache_info()
            print(f"Surface cache hits  : {cache.hits}")
            print(f"Surface cache misses: {cache.misses}")
        print("============================================\n", flush=True)

    
    def simulate(self, particles, duration, dt=None, start_time_index=0):
        if self.terrain is None:
            raise RuntimeError("TerrainService wajib tersedia.")
        if not particles:
            raise ValueError("Particles tidak boleh kosong.")

        params = self.turbulence.parameters() if self.turbulence else None
        duration = self._number(duration, "Duration", minimum=0.0)
        dt = self._number(
            params["timestep"] if dt is None and params
            else 71.25 if dt is None else dt,
            "Timestep",
            minimum=0.0
        )

        if duration <= 0 or dt <= 0:
            raise ValueError("Duration dan timestep harus positif.")

        info = self.meteorology.time_information()
        times = info.get("elapsed_seconds", [])
        start_time_index = int(start_time_index)

        if not times or not 0 <= start_time_index < len(times):
            raise ValueError("Time index meteorologi tidak valid.")

        start_time = float(times[start_time_index])

        if start_time + duration > float(times[-1]) + 1e-6:
            raise ValueError("Durasi simulasi melewati cakupan GFS.")

        rng = self.particle_simulation.rng
        steps = math.ceil(duration / dt)
        trajectories = []

        self._perf = {
            "start": time.perf_counter(),
            "transport_iterations": 0,
            "deposition_substeps": 0,
            "dem_crossing_calls": 0,
            "dem_crossing_time": 0.0,
            "terrain_calls": 0,
            "terrain_time": 0.0,
            "surface_calls": 0,
            "surface_time": 0.0,
            "landcover_calls": 0,
            "landcover_time": 0.0,
            "particle_step_calls": 0,
            "particle_step_time": 0.0,
            "event_rate_calls": 0,
            "event_rate_time": 0.0,
            "deposition_loop_time": 0.0
        }

        print("\n========== DISPERSION START ==========")
        print(f"Particles : {len(particles)}")
        print(f"Duration  : {duration:.2f} s")
        print(f"Timestep  : {dt:.2f} s")
        print(f"Max steps : {steps}")
        print("Mode      : FAST")
        print("======================================\n", flush=True)

        try:
            for particle_index, p in enumerate(particles, start=1):
                lat = float(p["latitude"])
                lon = float(p["longitude"])
                alt = float(p["altitude"])
                ground = self._terrain_elevation(lat, lon)

                if alt < ground:
                    raise ValueError("Partikel dimulai di bawah DEM.")

                release_time = self._number(
                    p.get("release_time", 0.0),
                    "Release time",
                    minimum=0.0
                )

                if release_time >= duration:
                    raise ValueError("Release time di luar durasi simulasi.")

                elapsed = release_time
                status = "active"
                deposition = None
                fluct = [0.0, 0.0, 0.0]
                settling = float(p.get("settling_velocity", 0.0))

                hazard_accum = 0.0
                hazard_threshold = (
                    -math.log1p(-float(rng.random()))
                    if self.deposition else None
                )

                initial_time_index = self.meteorology.time_index_at_elapsed(
                    start_time + elapsed
                )

                trajectory = [{
                    "time": elapsed,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "u": 0.0,
                    "v": 0.0,
                    "vertical": 0.0,
                    "time_index": initial_time_index
                }]

                while elapsed < duration - 1e-9:
                    self._perf["transport_iterations"] += 1

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

                    wind_time = start_time + elapsed

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

                    start_lat = lat
                    start_lon = lon
                    start_alt = alt

                    started = time.perf_counter()
                    try:
                        state = self.particle_simulation.step(
                            latitude=start_lat,
                            longitude=start_lon,
                            altitude=start_alt,
                            dt=step_dt,
                            time_index=float(wind_time),
                            settling_velocity=settling,
                            turbulence_velocity=fluct
                        )
                    finally:
                        self._perf["particle_step_time"] += (
                            time.perf_counter() - started
                        )
                        self._perf["particle_step_calls"] += 1

                    lat = float(state["latitude"])
                    lon = float(state["longitude"])
                    alt = float(state["altitude"])
                    ground = self._terrain_elevation(lat, lon)
                    end_time = elapsed + step_dt

                    if alt <= ground:
                        alt = ground
                        status = "deposited"
                        deposition = self._record(
                            p, lat, lon, alt, end_time,
                            "gravitational",
                            vd=settling
                        )

                    elif self.deposition:
                        deposition_start = time.perf_counter()

                        try:
                            height_agl = max(0.0, alt - ground)
                            near_surface = (
                                height_agl <= self.deposition.layer_height
                            )

                            started = time.perf_counter()
                            try:
                                conditions = self._surface_conditions(
                                    lat,
                                    lon,
                                    alt,
                                    self.meteorology.time_index_at_elapsed(
                                        start_time + end_time
                                    ),
                                    elapsed_seconds=start_time + end_time,
                                    include_landcover=near_surface
                                )
                            finally:
                                self._perf["surface_time"] += (
                                    time.perf_counter() - started
                                )
                                self._perf["surface_calls"] += 1

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
                                        "Roughness ESA tidak tersedia."
                                    )

                            started = time.perf_counter()
                            try:
                                rates = self.deposition.event_rates(
                                    particle=p,
                                    altitude=height_agl,
                                    friction_velocity=friction,
                                    roughness_length=roughness,
                                    rain_rate=rain,
                                    rh=rh
                                )
                            finally:
                                self._perf["event_rate_time"] += (
                                    time.perf_counter() - started
                                )
                                self._perf["event_rate_calls"] += 1

                            dry_rate = float(rates["dry_rate"])
                            wet_rate = float(rates["wet_rate"])
                            total_rate = dry_rate + wet_rate

                            if not all(
                                math.isfinite(x) and x >= 0
                                for x in (dry_rate, wet_rate)
                            ):
                                raise ValueError(
                                    "Laju deposisi tidak valid."
                                )

                            previous_hazard = hazard_accum
                            hazard_accum += total_rate * step_dt
                            self._perf["deposition_substeps"] += 1

                            if (
                                total_rate > 0
                                and hazard_accum >= hazard_threshold
                            ):
                                event_dt = min(
                                    step_dt,
                                    max(
                                        0.0,
                                        (hazard_threshold - previous_hazard)
                                        / total_rate
                                    )
                                )

                                mechanism = (
                                    "wet"
                                    if float(rng.random())
                                    < wet_rate / total_rate
                                    else "dry"
                                )

                                if event_dt > 0:
                                    started = time.perf_counter()
                                    try:
                                        event_state = self.particle_simulation.step(
                                            latitude=start_lat,
                                            longitude=start_lon,
                                            altitude=start_alt,
                                            dt=event_dt,
                                            time_index=float(wind_time),
                                            settling_velocity=settling,
                                            turbulence_velocity=fluct
                                        )
                                    finally:
                                        self._perf["particle_step_time"] += (
                                            time.perf_counter() - started
                                        )
                                        self._perf["particle_step_calls"] += 1

                                    lat = float(event_state["latitude"])
                                    lon = float(event_state["longitude"])
                                    alt = float(event_state["altitude"])
                                    state = event_state
                                else:
                                    lat = start_lat
                                    lon = start_lon
                                    alt = start_alt

                                ground = self._terrain_elevation(lat, lon)
                                end_time = elapsed + event_dt
                                status = "deposited"

                                dry_hazard = dry_rate * event_dt
                                wet_hazard = wet_rate * event_dt

                                deposition = self._record(
                                    p,
                                    lat,
                                    lon,
                                    alt,
                                    end_time,
                                    mechanism,
                                    probability=-math.expm1(-hazard_threshold),
                                    pdry=-math.expm1(-dry_hazard),
                                    pwet=-math.expm1(-wet_hazard),
                                    vd=rates["vd"],
                                    ra=rates["ra"],
                                    rs=rates["rs"],
                                    relative_humidity=rh,
                                    rain_rate_mm_hour=rain,
                                    scavenging_type=(
                                        self.deposition.scavenging_type(rh)
                                        if rain > 0 else None
                                    ),
                                    scavenging_coefficient=wet_rate,
                                    friction_velocity=rates["friction_velocity"],
                                    roughness_length=rates["roughness_length"],
                                    landcover_class=conditions.get(
                                        "landcover_class"
                                    ),
                                    landcover_name=conditions.get(
                                        "landcover_name"
                                    ),
                                    landcover_tile=conditions.get(
                                        "landcover_tile"
                                    )
                                )
                        finally:
                            self._perf["deposition_loop_time"] += (
                                time.perf_counter() - deposition_start
                            )

                    elapsed = (
                        end_time
                        if status == "deposited"
                        else min(duration, elapsed + step_dt)
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
                    **p,
                    "trajectory": trajectory,
                    "status": status,
                    "deposition": deposition
                })

                progress_interval = max(1, len(particles) // 10)

                if (
                    particle_index % progress_interval == 0
                    or particle_index == len(particles)
                ):
                    elapsed_perf = (
                        time.perf_counter() - self._perf["start"]
                    )

                    print(
                        f"[DISPERSION] "
                        f"{particle_index}/{len(particles)} | "
                        f"elapsed={elapsed_perf:.2f}s | "
                        f"steps={self._perf['transport_iterations']} | "
                        f"RK4={self._perf['particle_step_time']:.2f}s | "
                        f"surface={self._perf['surface_time']:.2f}s",
                        flush=True
                    )

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

        finally:
            self._print_performance(len(particles), steps)
            self._perf = None

