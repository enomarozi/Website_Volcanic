class SimulationService:

    def __init__(self, meteorology, turbulence, particle_simulation):
        self.meteorology = meteorology
        self.turbulence = turbulence
        self.particle_simulation = particle_simulation

    def run(self, latitude, longitude, altitude, particle_radius, particle_density, particle_count=100, duration=3600, dt=60, time_index=0):
        if particle_count <= 0:
            raise ValueError("Particle count must be greater than zero.")

        if duration <= 0:
            raise ValueError("Duration must be greater than zero.")

        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")

        particle = self.particle_simulation.particle.initialize_particle(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            particle_radius=particle_radius,
            particle_density=particle_density
        )

        states = []
        total_steps = int(duration / dt)

        for particle_id in range(particle_count):
            state = {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "altitude": float(altitude),
                "turbulent_u": 0.0,
                "turbulent_v": 0.0,
                "turbulent_w": 0.0
            }

            for step in range(total_steps + 1):
                states.append({
                    "particle_id": particle_id,
                    "time": step * dt,
                    "latitude": state["latitude"],
                    "longitude": state["longitude"],
                    "altitude": state["altitude"]
                })

                if step == total_steps:
                    break

                velocity = self.particle_simulation.velocity(
                    latitude=state["latitude"],
                    longitude=state["longitude"],
                    altitude=state["altitude"],
                    time_index=time_index,
                    turbulent_u=state["turbulent_u"],
                    turbulent_v=state["turbulent_v"],
                    turbulent_w=state["turbulent_w"]
                )

                position = self.particle_simulation.update_position(
                    latitude=state["latitude"],
                    longitude=state["longitude"],
                    altitude=state["altitude"],
                    velocity_u=velocity["u"],
                    velocity_v=velocity["v"],
                    velocity_w=velocity["w"],
                    settling_velocity=particle["settling_velocity"],
                    dt=dt
                )

                state["latitude"] = position["latitude"]
                state["longitude"] = position["longitude"]
                state["altitude"] = position["altitude"]

        return {
            "particle_count": particle_count,
            "duration": duration,
            "dt": dt,
            "settling_velocity": particle["settling_velocity"],
            "trajectories": states
        }