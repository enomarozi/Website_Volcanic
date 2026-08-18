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
            self.meteorology
            .wind_at_altitude(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                time_index=time_index
            )
        )

        return {
            "u": float(
                wind["u"]
            ),
            "v": float(
                wind["v"]
            )
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

        u = velocity["u"]
        v = velocity["v"]

        earth_radius = 6371000.0

        latitude_rad = (
            latitude * 3.141592653589793
            / 180.0
        )

        meters_per_degree_lat = (
            111320.0
        )

        meters_per_degree_lon = (
            111320.0
            * max(
                0.01,
                abs(
                    __import__(
                        "math"
                    ).cos(
                        latitude_rad
                    )
                )
            )
        )

        delta_latitude = (
            v * dt
            / meters_per_degree_lat
        )

        delta_longitude = (
            u * dt
            / meters_per_degree_lon
        )

        vertical = (
            -settling_velocity
        )

        new_latitude = (
            latitude + delta_latitude
        )

        new_longitude = (
            longitude + delta_longitude
        )

        new_altitude = max(
            0.0,
            altitude + (
                vertical * dt
            )
        )

        return {
            "latitude": new_latitude,
            "longitude": new_longitude,
            "altitude": new_altitude,
            "u": u,
            "v": v,
            "vertical": vertical,
            "time_index": time_index
        }