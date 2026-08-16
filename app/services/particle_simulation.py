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
        time_index
    ):
        wind = (
            self.meteorology.wind_at_altitude(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                time_index=time_index
            )
        )

        return {
            "u": wind["u"],
            "v": wind["v"],
            "vertical": 0.0
        }

    def step(
        self,
        latitude,
        longitude,
        altitude,
        dt,
        time_index,
        settling_velocity
    ):
        velocity = self.velocity(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )

        earth_radius = 6371000.0

        d_lat = (
            velocity["v"]
            * dt
            / earth_radius
        ) * (
            180.0 / math.pi
        )

        cos_latitude = math.cos(
            math.radians(
                latitude
            )
        )

        if abs(
            cos_latitude
        ) < 1e-8:
            cos_latitude = 1e-8

        d_lon = (
            velocity["u"]
            * dt
            / (
                earth_radius
                * cos_latitude
            )
        ) * (
            180.0 / math.pi
        )

        new_altitude = (
            altitude
            - settling_velocity * dt
        )

        new_altitude = max(
            new_altitude,
            0.0
        )

        return {
            "latitude": latitude + d_lat,
            "longitude": longitude + d_lon,
            "altitude": new_altitude,
            "u": velocity["u"],
            "v": velocity["v"],
            "vertical": -settling_velocity,
            "time_index": int(
                time_index
            )
        }