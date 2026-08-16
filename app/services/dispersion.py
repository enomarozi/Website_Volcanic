import math


class DispersionService:

    def __init__(
        self,
        particle_simulation,
        meteorology
    ):
        self.particle_simulation = particle_simulation
        self.meteorology = meteorology

        self.time_information = (
            self.meteorology.time_information()
        )

    def _get_time_index(
        self,
        elapsed_time,
        initial_time_index
    ):
        intervals = (
            self.time_information[
                "intervals_seconds"
            ]
        )

        if not intervals:
            return initial_time_index

        accumulated_time = 0.0
        current_index = initial_time_index

        for index in range(
            initial_time_index,
            len(intervals)
        ):
            accumulated_time += (
                intervals[index]
            )

            if elapsed_time < accumulated_time:
                return current_index

            current_index += 1

        return min(
            current_index,
            self.time_information[
                "count"
            ] - 1
        )

    def _validate_particle(
        self,
        particle
    ):
        required = [
            "id",
            "class",
            "radius",
            "mass",
            "latitude",
            "longitude",
            "altitude",
            "settling_velocity"
        ]

        missing = [
            key
            for key in required
            if key not in particle
        ]

        if missing:
            raise ValueError(
                "Particle is missing fields: "
                + ", ".join(missing)
            )

    def simulate(
        self,
        particles,
        duration,
        dt,
        time_index
    ):
        if not particles:
            raise ValueError(
                "Particles cannot be empty."
            )

        if duration <= 0:
            raise ValueError(
                "Duration must be greater than zero."
            )

        if dt <= 0:
            raise ValueError(
                "Time step must be greater than zero."
            )

        if time_index < 0:
            raise ValueError(
                "time_index cannot be negative."
            )

        meteorology_count = (
            self.time_information[
                "count"
            ]
        )

        if time_index >= meteorology_count:
            raise ValueError(
                "Initial time_index is outside "
                "meteorological data."
            )

        for particle in particles:
            self._validate_particle(
                particle
            )

        self.meteorology.simulation_time_range(
            time_index=time_index,
            duration=duration
        )

        steps = math.ceil(
            duration / dt
        )

        trajectories = []

        for particle in particles:
            current_latitude = float(
                particle["latitude"]
            )

            current_longitude = float(
                particle["longitude"]
            )

            current_altitude = max(
                float(
                    particle["altitude"]
                ),
                0.0
            )

            settling_velocity = max(
                float(
                    particle[
                        "settling_velocity"
                    ]
                ),
                0.0
            )

            trajectory = [{
                "step": 0,
                "time": 0.0,
                "time_index": time_index,
                "latitude": current_latitude,
                "longitude": current_longitude,
                "altitude": current_altitude,
                "u": 0.0,
                "v": 0.0,
                "vertical": 0.0
            }]

            elapsed_time = 0.0

            for step in range(
                1,
                steps + 1
            ):
                remaining_time = (
                    duration - elapsed_time
                )

                current_dt = min(
                    float(dt),
                    remaining_time
                )

                if current_dt <= 0:
                    break

                current_time_index = (
                    self._get_time_index(
                        elapsed_time=elapsed_time,
                        initial_time_index=time_index
                    )
                )

                state = (
                    self.particle_simulation.step(
                        latitude=current_latitude,
                        longitude=current_longitude,
                        altitude=current_altitude,
                        dt=current_dt,
                        time_index=current_time_index,
                        settling_velocity=settling_velocity
                    )
                )

                elapsed_time += current_dt

                current_latitude = float(
                    state["latitude"]
                )

                current_longitude = float(
                    state["longitude"]
                )

                current_altitude = max(
                    float(
                        state["altitude"]
                    ),
                    0.0
                )

                trajectory.append({
                    "step": step,
                    "time": elapsed_time,
                    "time_index": current_time_index,
                    "latitude": current_latitude,
                    "longitude": current_longitude,
                    "altitude": current_altitude,
                    "u": float(
                        state["u"]
                    ),
                    "v": float(
                        state["v"]
                    ),
                    "vertical": float(
                        state["vertical"]
                    )
                })

            trajectories.append({
                "particle_id": particle["id"],
                "class": particle["class"],
                "radius": particle["radius"],
                "mass": particle["mass"],
                "settling_velocity": settling_velocity,
                "initial_latitude": float(
                    particle["latitude"]
                ),
                "initial_longitude": float(
                    particle["longitude"]
                ),
                "initial_altitude": float(
                    particle["altitude"]
                ),
                "trajectory": trajectory
            })

        return {
            "particle_count": len(
                trajectories
            ),
            "duration": duration,
            "dt": dt,
            "steps": steps,
            "initial_time_index": time_index,
            "meteorology": {
                "count": self.time_information[
                    "count"
                ],
                "intervals_seconds": (
                    self.time_information[
                        "intervals_seconds"
                    ]
                ),
                "times": [
                    str(time)
                    for time in self.time_information[
                        "times"
                    ]
                ]
            },
            "trajectories": trajectories
        }