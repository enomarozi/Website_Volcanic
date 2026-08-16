import numpy as np


class EruptionService:

    def __init__(
        self,
        air_density: float = 1.225,
        gravity: float = 9.81,
        alpha: float = 0.1,
        beta: float = 0.5,
        z1: float = 2.8
    ):
        self.air_density = air_density
        self.gravity = gravity
        self.alpha = alpha
        self.beta = beta
        self.z1 = z1

    def mean_wind_speed(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        if u.size == 0 or v.size == 0:
            raise ValueError("Wind profile cannot be empty.")

        speed = np.sqrt(u ** 2 + v ** 2)

        return float(np.mean(speed))

    def brunt_vaisala_frequency(self, potential_temperature, altitude):
        theta = np.asarray(potential_temperature, dtype=float)
        altitude = np.asarray(altitude, dtype=float)

        if len(theta) < 2:
            raise ValueError("At least two atmospheric levels are required.")

        theta_mean = float(np.mean(theta))

        dz = float(altitude[-1] - altitude[0])

        if dz == 0:
            raise ValueError("Atmospheric altitude range cannot be zero.")

        dtheta_dz = float(
            (theta[-1] - theta[0]) / dz
        )

        n_squared = (
            self.gravity / theta_mean
        ) * dtheta_dz

        n_squared = max(n_squared, 0.0)

        return {
            "theta_mean": theta_mean,
            "dtheta_dz": dtheta_dz,
            "n_squared": float(n_squared),
            "n": float(np.sqrt(n_squared))
        }

    def mass_eruption_rate(
        self,
        eruption_height: float,
        brunt_vaisala: float,
        mean_wind_speed: float
    ):
        H = float(eruption_height)
        N = float(brunt_vaisala)
        v_bar = float(mean_wind_speed)

        if H <= 0:
            raise ValueError("Eruption height must be greater than zero.")

        if N <= 0:
            raise ValueError("Brunt-Vaisala frequency must be greater than zero.")

        if v_bar < 0:
            raise ValueError("Mean wind speed cannot be negative.")

        buoyancy_term = (
            (
                2.0
                * self.alpha ** 2
                * N ** 3
            )
            / (
                5.0
                * self.z1 ** 4
            )
        ) * H ** 4

        wind_term = (
            (
                self.beta ** 2
                * N ** 2
                * v_bar
            )
            / 6.0
        ) * H ** 3

        mer = (
            np.pi
            * self.air_density
            / self.gravity
            * (buoyancy_term + wind_term)
        )

        return {
            "buoyancy_term": float(buoyancy_term),
            "wind_term": float(wind_term),
            "mer": float(mer)
        }

    def total_mass(
        self,
        mer: float,
        eruption_duration: float
    ):
        if mer < 0:
            raise ValueError("MER cannot be negative.")

        if eruption_duration <= 0:
            raise ValueError(
                "Eruption duration must be greater than zero."
            )

        return float(mer * eruption_duration)

    def calculate(
        self,
        atmospheric_profile,
        eruption_height: float,
        eruption_duration: float
    ):
        profile = atmospheric_profile["profile"]

        u = np.asarray(
            [item["u"] for item in profile],
            dtype=float
        )

        v = np.asarray(
            [item["v"] for item in profile],
            dtype=float
        )

        theta = np.asarray(
            [
                item["potential_temperature"]
                for item in profile
            ],
            dtype=float
        )

        altitude = np.asarray(
            [
                item["altitude"]
                for item in profile
            ],
            dtype=float
        )

        mean_wind = self.mean_wind_speed(u, v)

        stability = self.brunt_vaisala_frequency(
            theta,
            altitude
        )

        mer_result = self.mass_eruption_rate(
            eruption_height=eruption_height,
            brunt_vaisala=stability["n"],
            mean_wind_speed=mean_wind
        )

        total_mass = self.total_mass(
            mer=mer_result["mer"],
            eruption_duration=eruption_duration
        )

        return {
            "eruption_height": float(eruption_height),
            "eruption_duration": float(eruption_duration),
            "mean_wind_speed": mean_wind,
            "theta_mean": stability["theta_mean"],
            "dtheta_dz": stability["dtheta_dz"],
            "n_squared": stability["n_squared"],
            "brunt_vaisala": stability["n"],
            "buoyancy_term": mer_result["buoyancy_term"],
            "wind_term": mer_result["wind_term"],
            "mer": mer_result["mer"],
            "total_mass": total_mass
        }