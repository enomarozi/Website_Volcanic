class DispersionService:
    def __init__(self, particle_simulation, meteorology):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology

    def simulate(self, particles, duration, dt, start_time_index=0):
        if not particles:
            raise ValueError("Particles cannot be empty.")
        if duration <= 0:
            raise ValueError("Duration must be greater than zero.")
        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")
        if duration < dt:
            raise ValueError("Duration must be greater than or equal to dt.")

        time_information = self.meteorology.time_information()
        meteorology_times = time_information.get("elapsed_seconds", [])
        meteorology_count = len(meteorology_times)

        if meteorology_count == 0:
            raise ValueError("Meteorological time information is empty.")

        if start_time_index < 0 or start_time_index >= meteorology_count:
            raise ValueError(f"Invalid start_time_index: {start_time_index}")

        steps = int(duration / dt)
        trajectories = []

        for particle in particles:
            current_latitude = float(particle["latitude"])
            current_longitude = float(particle["longitude"])
            current_altitude = float(particle["altitude"])
            settling_velocity = float(particle.get("settling_velocity", 0.0))
            particle_id = particle.get("id", len(trajectories) + 1)
            particle_class = particle.get("class", "unknown")
            radius = particle.get("radius", 0.0)
            mass = particle.get("mass", 0.0)

            trajectory = [{
                "time": 0.0,
                "latitude": current_latitude,
                "longitude": current_longitude,
                "altitude": current_altitude,
                "u": 0.0,
                "v": 0.0,
                "vertical": 0.0,
                "time_index": start_time_index
            }]

            for step in range(1, steps + 1):
                elapsed_time = step * dt
                time_index = self.meteorology.time_index_at_elapsed(
                    elapsed_seconds=meteorology_times[start_time_index] + elapsed_time
                )

                state = self.particle_simulation.step(
                    latitude=current_latitude,
                    longitude=current_longitude,
                    altitude=current_altitude,
                    dt=dt,
                    time_index=time_index,
                    settling_velocity=settling_velocity
                )

                current_latitude = float(state["latitude"])
                current_longitude = float(state["longitude"])
                current_altitude = float(state["altitude"])

                trajectory.append({
                    "time": elapsed_time,
                    "latitude": current_latitude,
                    "longitude": current_longitude,
                    "altitude": current_altitude,
                    "u": float(state.get("u", 0.0)),
                    "v": float(state.get("v", 0.0)),
                    "vertical": float(state.get("vertical", 0.0)),
                    "time_index": int(state.get("time_index", time_index))
                })

            trajectories.append({
                "particle_id": particle_id,
                "class": particle_class,
                "radius": radius,
                "mass": mass,
                "settling_velocity": settling_velocity,
                "trajectory": trajectory
            })

        return {
            "particle_count": len(trajectories),
            "total_particles": sum(int(particle.get("count", 1)) for particle in particles),
            "duration": duration,
            "dt": dt,
            "steps": steps,
            "start_time_index": start_time_index,
            "meteorology": {
                "time_count": meteorology_count,
                "interval_seconds": time_information.get("interval_seconds"),
                "interval_hours": time_information.get("interval_hours"),
                "times": [str(time) for time in time_information.get("times", [])],
                "elapsed_seconds": [float(value) for value in meteorology_times]
            },
            "trajectories": trajectories
        }