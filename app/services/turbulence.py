import numpy as np


class TurbulenceService:

    def __init__(self, gravity=9.80665, min_diffusion=0.1):
        self.gravity = gravity
        self.min_diffusion = min_diffusion

    def vertical_gradient(self, values, altitude):
        v, z = np.asarray(values, float), np.asarray(altitude, float)
        if v.size != z.size or v.size < 2:
            raise ValueError("Values and altitude must contain >= 2 equal levels.")
        order = np.argsort(z)
        return np.gradient(v[order], z[order])

    def calculate_stability(self, potential_temperature, altitude):
        theta = np.asarray(potential_temperature, float)
        z = np.asarray(altitude, float)
        dtheta_dz = self.vertical_gradient(theta, z)
        n2 = np.maximum(self.gravity / np.maximum(theta, 1e-12) * dtheta_dz, 0)
        return {"dtheta_dz": dtheta_dz, "n_squared": n2, "n": np.sqrt(n2)}

    def coriolis_parameter(self, latitude):
        return float(2 * 7.2921159e-5 * np.sin(np.deg2rad(latitude)))

    def diffusion_coefficients(
        self,
        wind_speed,
        altitude,
        brunt_vaisala=0.0,
        boundary_layer=1000.0
    ):
        """
        Simple atmospheric turbulent diffusion parameterization.

        K values are in m2/s.
        This is the numerical dispersion component and can later
        be replaced by a more specific PDF parameterization.
        """
        u = max(abs(float(wind_speed)), 0.1)
        z = max(float(altitude), 0.0)
        n = max(float(brunt_vaisala), 0.0)

        stability = 1.0 / (1.0 + n * 100.0)
        mixing = max(0.05, 1.0 - z / max(boundary_layer, 1.0))

        kh = max(self.min_diffusion, 0.10 * u * 1000.0 * stability)
        kv = max(self.min_diffusion, 0.10 * u * 100.0 * mixing * stability)

        return {
            "horizontal": float(kh),
            "vertical": float(kv)
        }

    def turbulent_velocity(self, diffusion, dt, rng=None):
        """
        Random turbulent velocity generated from diffusion.
        """
        rng = rng or np.random.default_rng()
        k = max(float(diffusion), 0.0)
        dt = max(float(dt), 1e-12)
        displacement = rng.normal(0.0, np.sqrt(2.0 * k * dt))
        return float(displacement / dt)