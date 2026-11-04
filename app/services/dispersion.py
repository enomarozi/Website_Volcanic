
import math

class DispersionService:
    def __init__(self, particle_simulation, meteorology, deposition=None, turbulence=None, surface_meteorology=None):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology
        self.deposition = deposition
        self.turbulence = turbulence
        self.surface_meteorology = surface_meteorology

    def _surface_conditions(self, lat, lon, alt, time_index):
        result = {}

        for name in ("conditions_at_position", "weather_at_position", "sample_at_position"):
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
            try:
                surface_index = self.surface_meteorology.time_index_at_elapsed(
                    self.meteorology.time_information()["elapsed_seconds"][time_index]
                )
                surface = self.surface_meteorology.surface_profile(
                    lat, lon, surface_index
                )
                result.update(surface)
            except (ValueError, KeyError, IndexError, TypeError):
                pass

        if not result:
            try:
                value = self.meteorology.wind_at_altitude(lat, lon, alt, time_index)
                if isinstance(value, dict):
                    result.update(value)
            except Exception:
                pass

        return result

    def simulate(self, particles, duration, dt=None, start_time_index=0):
        duration = float(duration)

        if dt is None:
            dt = self.turbulence.parameters()["timestep"] if self.turbulence else 71.25

        dt = float(dt)

        if not particles or duration <= 0 or dt <= 0:
            raise ValueError("Particles, duration and dt must be valid.")

        info = self.meteorology.time_information()
        elapsed_times = info.get("elapsed_seconds", [])

        if not elapsed_times:
            raise ValueError("Meteorological time information is empty.")

        start_time_index = int(start_time_index)
        if not 0 <= start_time_index < len(elapsed_times):
            raise ValueError("start_time_index exceeds meteorological dataset.")

        rng = getattr(self.particle_simulation, "rng", None)
        steps = math.ceil(duration / dt)
        trajectories = []

        for p in particles:
            lat = float(p["latitude"])
            lon = float(p["longitude"])
            alt = float(p["altitude"])

            trajectory = [{
                "time": 0.0, "latitude": lat, "longitude": lon,
                "altitude": alt, "u": 0.0, "v": 0.0,
                "vertical": 0.0, "time_index": start_time_index
            }]

            status = "active"
            deposition = None
            fluct = [0.0, 0.0, 0.0]
            elapsed = 0.0

            while elapsed < duration - 1e-9:
                step_dt = min(dt, duration - elapsed)
                target = elapsed_times[start_time_index] + elapsed + step_dt
                ti = self.meteorology.time_index_at_elapsed(target)

                if self.turbulence:
                    params = self.turbulence.parameters()
                    fluct = [
                        self.turbulence.langevin_velocity(
                            fluct[i], params["sigma_u"], params["tau_l"], step_dt, rng
                        ) for i in range(3)
                    ]

                state = self.particle_simulation.step(
                    lat, lon, alt, step_dt, ti,
                    p.get("settling_velocity", 0.0), fluct
                )

                lat, lon, alt = state["latitude"], state["longitude"], state["altitude"]
                elapsed = min(duration, elapsed + step_dt)

                trajectory.append({
                    "time": elapsed, "latitude": lat, "longitude": lon,
                    "altitude": alt, "u": state["u"], "v": state["v"],
                    "vertical": state["vertical"], "time_index": ti
                })

                if self.deposition and alt <= 0.0 and str(
                    p.get("particle_class", p.get("class", ""))
                ).lower() == "coarse":
                    status = "deposited"
                    deposition = {
                        "latitude": lat, "longitude": lon, "altitude": 0.0,
                        "time": elapsed, "mass": float(p.get("mass", 0.0)),
                        "mechanism": "gravity", "probability": 1.0,
                        "dry_probability": 0.0, "wet_probability": 0.0,
                        "total_probability": 1.0,
                        "vd": float(p.get("settling_velocity", 0.0))
                    }
                    trajectory[-1]["altitude"] = 0.0
                    break

                if self.deposition and alt <= getattr(self.deposition, "layer_height", 100.0):
                    cond = self._surface_conditions(lat, lon, alt, ti)

                    rain = cond.get(
                        "precipitation_rate_mm_hour",
                        cond.get("rain_rate", cond.get("precipitation", cond.get("precip")))
                    )
                    rh = cond.get("relative_humidity", cond.get("rh", cond.get("r")))

                    dep = self.deposition.decide(
                        p, alt, step_dt, rain_rate=rain, rh=rh, rng=rng
                    )

                    if dep["deposited"]:
                        status = "deposited"
                        deposition = {
                            "latitude": lat, "longitude": lon,
                            "altitude": alt, "time": elapsed,
                            "mass": float(p.get("mass", 0.0)),
                            "mechanism": dep["mechanism"],
                            "probability": dep["probability"],
                            "dry_probability": dep.get("pdry", 0.0),
                            "wet_probability": dep.get("pwet", 0.0),
                            "total_probability": dep.get("probability", 0.0),
                            "vd": dep.get("vd", 0.0)
                        }
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
