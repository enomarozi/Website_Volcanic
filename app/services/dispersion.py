class DispersionService:

    def __init__(self, particle_simulation, meteorology):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology

    def simulate(
        self,
        particles,
        duration,
        dt,
        start_time_index=0
    ):
        if not particles:
            raise ValueError("Particles cannot be empty.")
        if duration <= 0 or dt <= 0:
            raise ValueError("Duration and dt must be greater than zero.")

        info = self.meteorology.time_information()
        times = info["elapsed_seconds"]

        if not times:
            raise ValueError("Meteorological time information is empty.")

        if not 0 <= start_time_index < len(times):
            raise ValueError("Invalid start_time_index.")

        steps = int(duration // dt)
        trajectories = []
        deposited_mass = 0.0

        for particle in particles:
            lat = float(particle["latitude"])
            lon = float(particle["longitude"])
            alt = float(particle["altitude"])
            mass = float(particle.get("mass", 0.0))
            settling = float(particle.get("settling_velocity", 0.0))

            trajectory = [{
                "time": 0.0,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "u": 0.0,
                "v": 0.0,
                "vertical": 0.0,
                "time_index": start_time_index,
                "deposited": False
            }]

            deposited = False
            deposition = None

            for step in range(1, steps + 1):
                elapsed = step * dt

                absolute_elapsed = (
                    times[start_time_index] + elapsed
                )

                time_index = self.meteorology.time_index_at_elapsed(
                    absolute_elapsed
                )

                if deposited:
                    trajectory.append({
                        **trajectory[-1],
                        "time": elapsed,
                        "deposited": True
                    })
                    continue

                state = self.particle_simulation.step(
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    dt=dt,
                    time_index=time_index,
                    settling_velocity=settling
                )

                lat = state["latitude"]
                lon = state["longitude"]
                alt = state["altitude"]

                deposited = state["deposited"]

                item = {
                    "time": elapsed,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "u": state["u"],
                    "v": state["v"],
                    "vertical": state["vertical"],
                    "diffusion_horizontal":
                        state["diffusion_horizontal"],
                    "diffusion_vertical":
                        state["diffusion_vertical"],
                    "time_index": state["time_index"],
                    "deposited": deposited
                }

                trajectory.append(item)

                if deposited:
                    deposition = {
                        "time": elapsed,
                        "latitude": lat,
                        "longitude": lon,
                        "mass": mass
                    }
                    deposited_mass += mass

            trajectories.append({
                "particle_id": particle.get("id"),
                "class": particle.get("class", "unknown"),
                "radius": float(particle.get("radius", 0)),
                "mass": mass,
                "settling_velocity": settling,
                "deposited": deposited,
                "deposition": deposition,
                "trajectory": trajectory
            })

        total_mass = sum(
            float(p.get("mass", 0.0))
            for p in particles
        )

        airborne_mass = max(
            0.0,
            total_mass - deposited_mass
        )

        return {
            "particle_count": len(trajectories),
            "total_particles": len(particles),
            "duration": float(duration),
            "dt": float(dt),
            "steps": steps,
            "start_time_index": int(start_time_index),
            "total_mass": total_mass,
            "deposited_mass": deposited_mass,
            "airborne_mass": airborne_mass,
            "mass_error": total_mass - (
                deposited_mass + airborne_mass
            ),
            "meteorology": {
                "time_count": len(times),
                "interval_seconds":
                    info["interval_seconds"],
                "interval_hours":
                    info["interval_hours"],
                "times": [
                    str(t) for t in info["times"]
                ],
                "elapsed_seconds": [
                    float(t) for t in times
                ]
            },
            "trajectories": trajectories
        }