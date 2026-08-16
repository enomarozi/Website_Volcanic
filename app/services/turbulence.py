import numpy as np


class TurbulenceService:

    def __init__(
        self,
        gravity: float = 9.81
    ):
        self.gravity = gravity

    def vertical_gradient(
        self,
        values,
        altitude
    ):
        values = np.asarray(
            values,
            dtype=float
        )

        altitude = np.asarray(
            altitude,
            dtype=float
        )

        if len(values) != len(altitude):
            raise ValueError(
                "Values and altitude must have the same length."
            )

        if len(values) < 2:
            raise ValueError(
                "At least two atmospheric levels are required."
            )

        return np.gradient(
            values,
            altitude
        )

    def calculate_stability(
        self,
        potential_temperature,
        altitude
    ):
        theta = np.asarray(
            potential_temperature,
            dtype=float
        )

        altitude = np.asarray(
            altitude,
            dtype=float
        )

        dtheta_dz = self.vertical_gradient(
            theta,
            altitude
        )

        n_squared = (
            self.gravity / theta
        ) * dtheta_dz

        n_squared = np.maximum(
            n_squared,
            0.0
        )

        return {
            "dtheta_dz": dtheta_dz,
            "n_squared": n_squared,
            "n": np.sqrt(n_squared)
        }

    def coriolis_parameter(
        self,
        latitude: float
    ):
        omega = 7.2921159e-5

        return (
            2.0
            * omega
            * np.sin(
                np.deg2rad(latitude)
            )
        )