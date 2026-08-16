import math


class ParticleSimulationService:

    def __init__(
        self,
        meteorology,
        turbulence,
        particle
    ):
        self.meteorology = meteorology
        self.turbulence = turbulence
        self.particle = particle

    def velocity(
        self,
        latitude,
        longitude,
        altitude,
        time_index=0
    ):
        wind = self.meteorology.wind_at_altitude(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )

        return {
            "u": float(wind["u"]),
            "v": float(wind["v"]),
            "vertical": 0.0
        }

    def step(
        self,
        latitude,
        longitude,
        altitude,
        dt=60,
        time_index=0,
        settling_velocity=0.0
    ):
        if dt <= 0:
            raise ValueError(
                "Time step must be greater than zero."
            )

        velocity = self.velocity(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )

        earth_radius = 6371000.0

        latitude_radians = math.radians(
            latitude
        )

        delta_latitude = (
            velocity["v"]
            * dt
            / earth_radius
        ) * (
            180.0 / math.pi
        )

        cos_latitude = max(
            abs(
                math.cos(
                    latitude_radians
                )
            ),
            1e-8
        )

        delta_longitude = (
            velocity["u"]
            * dt
            / (
                earth_radius
                * cos_latitude
            )
        ) * (
            180.0 / math.pi
        )

        settling_velocity = max(
            float(settling_velocity),
            0.0
        )

        new_latitude = (
            latitude
            + delta_latitude
        )

        new_longitude = (
            longitude
            + delta_longitude
        )

        new_longitude = (
            (new_longitude + 180.0)
            % 360.0
        ) - 180.0

        new_altitude = (
            altitude
            - settling_velocity * dt
        )

        new_altitude = max(
            new_altitude,
            0.0
        )

        return {
            "latitude": new_latitude,
            "longitude": new_longitude,
            "altitude": new_altitude,
            "u": velocity["u"],
            "v": velocity["v"],
            "vertical": -settling_velocity
        }