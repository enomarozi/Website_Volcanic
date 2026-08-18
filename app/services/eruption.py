import math


class EruptionService:

    def calculate(
        self,
        atmospheric_profile,
        eruption_height,
        eruption_duration
    ):

        if not atmospheric_profile:
            raise ValueError(
                "Atmospheric profile cannot be empty."
            )

        if eruption_height <= 0:
            raise ValueError(
                "Eruption height must be greater than zero."
            )

        if eruption_duration <= 0:
            raise ValueError(
                "Eruption duration must be greater than zero."
            )

        profile = (
            atmospheric_profile[
                "profile"
            ]
        )

        selected = [
            item
            for item in profile
            if item["altitude"]
            <= eruption_height
        ]

        if not selected:
            selected = profile

        wind_speeds = [
            math.sqrt(
                item["u"] ** 2
                +
                item["v"] ** 2
            )
            for item in selected
        ]

        theta_values = [
            item["potential_temperature"]
            for item in selected
        ]

        brunt_values = [
            item["brunt_vaisala"]
            for item in selected
        ]

        altitudes = [
            item["altitude"]
            for item in selected
        ]

        mean_wind_speed = (
            sum(wind_speeds)
            / len(wind_speeds)
        )

        theta_mean = (
            sum(theta_values)
            / len(theta_values)
        )

        brunt_vaisala = (
            sum(brunt_values)
            / len(brunt_values)
        )

        if len(selected) >= 2:

            dz = (
                altitudes[-1]
                - altitudes[0]
            )

            dtheta = (
                theta_values[-1]
                - theta_values[0]
            )

            if abs(dz) > 1e-12:

                dtheta_dz = (
                    dtheta / dz
                )

            else:

                dtheta_dz = 0.0

        else:

            dtheta_dz = 0.0

        density = 1.225

        entrainment = 0.1

        mer = (
            density
            * math.pi
            * (
                max(
                    eruption_height,
                    1.0
                )
                ** 2
            )
            * entrainment
            * max(
                mean_wind_speed,
                1.0
            )
        )

        total_mass = (
            mer
            * eruption_duration
        )

        return {
            "eruption_height": float(
                eruption_height
            ),
            "eruption_duration": float(
                eruption_duration
            ),
            "mean_wind_speed": float(
                mean_wind_speed
            ),
            "theta_mean": float(
                theta_mean
            ),
            "dtheta_dz": float(
                dtheta_dz
            ),
            "brunt_vaisala": float(
                brunt_vaisala
            ),
            "mer": float(
                mer
            ),
            "total_mass": float(
                total_mass
            ),
            "profile_levels": len(
                selected
            )
        }