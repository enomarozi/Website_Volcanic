import math
import numpy as np


class ParticleSimulationService:

    def __init__(self, meteorology, turbulence, particle=None, seed=42):
        self.meteorology = meteorology
        self.turbulence = turbulence
        self.particle = particle
        self.rng = np.random.default_rng(seed)

    def velocity(self, latitude, longitude, altitude, time_index):
        w = self.meteorology.wind_at_altitude(
            latitude, longitude, altitude, time_index
        )
        return {
            "u": float(w["u"]),
            "v": float(w["v"]),
            "wind_speed": float(w["wind_speed"]),
            "brunt_vaisala": float(w.get("brunt_vaisala", 0.0))
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
        wind = self.velocity(
            latitude, longitude, altitude, time_index
        )

        u, v = wind["u"], wind["v"]

        diffusion = self.turbulence.diffusion_coefficients(
            wind_speed=wind["wind_speed"],
            altitude=altitude,
            brunt_vaisala=wind["brunt_vaisala"]
        )

        horizontal_sigma = math.sqrt(
            2.0 * diffusion["horizontal"] * dt
        )

        vertical_sigma = math.sqrt(
            2.0 * diffusion["vertical"] * dt
        )

        dx = u * dt + self.rng.normal(0.0, horizontal_sigma)
        dy = v * dt + self.rng.normal(0.0, horizontal_sigma)
        dz = (
            -abs(float(settling_velocity)) * dt
            + self.rng.normal(0.0, vertical_sigma)
        )

        lat = float(latitude) + dy / 111320.0

        lon_scale = max(
            abs(math.cos(math.radians(float(latitude)))),
            1e-8
        )

        lon = float(longitude) + dx / (111320.0 * lon_scale)
        altitude_new = max(0.0, float(altitude) + dz)

        if lon > 180:
            lon -= 360
        elif lon < -180:
            lon += 360

        return {
            "latitude": lat,
            "longitude": lon,
            "altitude": altitude_new,
            "u": u,
            "v": v,
            "vertical": float(dz / dt),
            "diffusion_horizontal": diffusion["horizontal"],
            "diffusion_vertical": diffusion["vertical"],
            "time_index": int(time_index),
            "deposited": altitude_new <= 0.0
        }