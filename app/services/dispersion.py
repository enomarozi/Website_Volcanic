class DispersionService:

    def __init__(self, particle_simulation):
        self.particle_simulation = particle_simulation

    def simulate(
        self,
        particles,
        duration=3600,
        dt=60,
        time_index=0
    ):
        if not particles:
            raise ValueError("Particles cannot be empty.")

        if duration <= 0:
            raise ValueError("Duration must be greater than zero.")

        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")

        steps = int(duration / dt)
        trajectories = []

        for particle in particles:
            current_latitude = particle["latitude"]
            current_longitude = particle["longitude"]
            current_altitude = particle["altitude"]

            settling_velocity = particle["settling_velocity"]

            trajectory = [{
                "time": 0.0,
                "latitude": current_latitude,
                "longitude": current_longitude,
                "altitude": current_altitude,
                "u": 0.0,
                "v": 0.0,
                "vertical": 0.0
            }]

            for step in range(1, steps + 1):
                state = self.particle_simulation.step(
                    latitude=current_latitude,
                    longitude=current_longitude,
                    altitude=current_altitude,
                    dt=dt,
                    time_index=time_index,
                    settling_velocity=settling_velocity
                )

                current_latitude = state["latitude"]
                current_longitude = state["longitude"]
                current_altitude = state["altitude"]

                trajectory.append({
                    "time": step * dt,
                    "latitude": current_latitude,
                    "longitude": current_longitude,
                    "altitude": current_altitude,
                    "u": state["u"],
                    "v": state["v"],
                    "vertical": state["vertical"]
                })

            trajectories.append({
                "particle_id": particle["id"],
                "class": particle["class"],
                "radius": particle["radius"],
                "mass": particle["mass"],
                "settling_velocity": settling_velocity,
                "trajectory": trajectory
            })

        return {
            "particle_count": len(trajectories),
            "duration": duration,
            "dt": dt,
            "steps": steps,
            "trajectories": trajectories
        }