
import math
import numpy as np


class TurbulenceService:
    def __init__(self):
        self.gravity = 9.81
        self.von_karman = 0.4
        self.ventilation_rate = 4000.0
        self.pbl_wind_speed = 7.018
        self.convective_velocity = 2.0
        self.sigma_factor = 0.6
        self.lagrangian_factor = 0.15
        self.timestep_factor = 1.0

    def vertical_gradient(self, values, altitude):
        values = np.asarray(values, dtype=float)
        z = np.asarray(altitude, dtype=float)

        if values.size != z.size or values.size < 2:
            raise ValueError("At least two atmospheric levels are required.")

        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(z)):
            raise ValueError("Atmospheric profile contains non-finite values.")

        if np.any(np.diff(z) <= 0):
            raise ValueError("Atmospheric altitudes must be strictly increasing.")

        return np.gradient(values, z)

    def calculate_stability(self, potential_temperature, altitude):
        theta = np.asarray(potential_temperature, dtype=float)
        grad = self.vertical_gradient(theta, altitude)
        n2 = np.maximum(
            self.gravity / np.maximum(theta, 1e-12) * grad,
            0.0
        )

        return {
            "dtheta_dz": grad,
            "n_squared": n2,
            "n": np.sqrt(n2)
        }

    def coriolis_parameter(self, latitude):
        return (
            2 * 7.2921159e-5 *
            math.sin(math.radians(float(latitude)))
        )

    def mixing_height(self, ventilation_rate=None, pbl_wind_speed=None, **_):
        return self.ventilation_rate / self.pbl_wind_speed

    def parameters(self, mixing_height=None, friction_velocity=None, stability=0.0):
        h = self.mixing_height()
        sigma = self.sigma_factor * self.convective_velocity
        tau = round(self.lagrangian_factor * h / sigma, 2)
        dt = round(self.timestep_factor * tau, 2)

        return {
            "mixing_height": round(h, 2),
            "sigma_u": sigma,
            "tau_l": tau,
            "timestep": dt,
            "diffusivity": sigma * sigma * tau,
            "ventilation_rate": self.ventilation_rate,
            "pbl_wind_speed": self.pbl_wind_speed,
            "convective_velocity": self.convective_velocity
        }

    def langevin_velocity(self, previous, sigma_u, tau_l, dt, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        previous = float(previous)
        sigma_u = max(float(sigma_u), 0.0)
        tau_l = max(float(tau_l), 1e-9)
        dt = max(float(dt), 0.0)

        decay = math.exp(-dt / tau_l)
        noise = sigma_u * math.sqrt(max(0.0, 1.0 - decay ** 2))

        return float(decay * previous + noise * rng.normal())

    def turbulent_displacement(self, fluctuation_velocity, dt):
        return float(fluctuation_velocity) * float(dt)
