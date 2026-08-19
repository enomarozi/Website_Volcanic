import math


class ParticleSimulationService:
    def __init__(self, meteorology, turbulence, particle):
        self.meteorology = meteorology
        self.turbulence = turbulence
        self.particle = particle

    def velocity(self, latitude, longitude, altitude, time_index):
        wind = self.meteorology.wind_at_altitude(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )
        return {"u": float(wind["u"]), "v": float(wind["v"])}

    def step(self, latitude, longitude, altitude, dt, time_index, settling_velocity):
        velocity = self.velocity(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )

        u = velocity["u"]
        v = velocity["v"]
        latitude_rad = math.radians(float(latitude))
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = meters_per_degree_lat * max(
            abs(math.cos(latitude_rad)), 1e-8
        )

        delta_latitude = v * dt / meters_per_degree_lat
        delta_longitude = u * dt / meters_per_degree_lon
        vertical = -float(settling_velocity)

        new_latitude = float(latitude + delta_latitude)
        new_longitude = float(longitude + delta_longitude)
        new_altitude = max(0.0, float(altitude + vertical * dt))

        if new_longitude > 180.0:
            new_longitude -= 360.0
        elif new_longitude < -180.0:
            new_longitude += 360.0

        return {
            "latitude": new_latitude,
            "longitude": new_longitude,
            "altitude": new_altitude,
            "u": u,
            "v": v,
            "vertical": vertical,
            "time_index": int(time_index)
        }