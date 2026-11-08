
import math
import numpy as np


class DepositionService:
    def __init__(
        self,
        gravity=9.81,
        von_karman=0.4,
        air_density=1.225,
        viscosity=1.81e-5,
        rain_a=8.4e-5,
        rain_b=0.79,
        rain_a_below_cloud=3.36e-4,
        rain_b_below_cloud=0.305,
        rh_threshold=80.0,
        reference_height=10.0,
        roughness_length=0.0133,
        friction_velocity=0.272,
        layer_height=100.0,
        surface_resistance_value=2.53,
        default_rain_rate=2.5
    ):
        self.g = float(gravity)
        self.kappa = float(von_karman)
        self.rho = float(air_density)
        self.viscosity = float(viscosity)

        self.rain_a = float(rain_a)
        self.rain_b = float(rain_b)
        self.rain_a_in_cloud = self.rain_a
        self.rain_b_in_cloud = self.rain_b
        self.rain_a_below_cloud = float(rain_a_below_cloud)
        self.rain_b_below_cloud = float(rain_b_below_cloud)

        self.rh_threshold = float(rh_threshold)
        self.reference_height = float(reference_height)
        self.roughness_length = float(roughness_length)
        self.friction_velocity = float(friction_velocity)
        self.layer_height = float(layer_height)

        self.surface_resistance_value = (
            None if surface_resistance_value is None
            else float(surface_resistance_value)
        )
        self.default_rain_rate = float(default_rain_rate)

        if self.g <= 0 or self.kappa <= 0 or self.rho <= 0:
            raise ValueError("Konstanta fisika harus lebih besar dari 0.")

        if self.viscosity <= 0:
            raise ValueError("Viscosity harus lebih besar dari 0.")

        if self.reference_height <= 0 or self.layer_height <= 0:
            raise ValueError("Reference height dan layer height harus positif.")

        if self.roughness_length <= 0:
            raise ValueError("Roughness length harus positif.")

        if not 0 <= self.rh_threshold <= 100:
            raise ValueError("RH threshold harus berada pada rentang 0-100%.")

        if min(
            self.rain_a_in_cloud,
            self.rain_a_below_cloud
        ) < 0:
            raise ValueError("Parameter A scavenging tidak boleh negatif.")

        if min(
            self.rain_b_in_cloud,
            self.rain_b_below_cloud
        ) < 0:
            raise ValueError("Eksponen scavenging tidak boleh negatif.")

    @staticmethod
    def _finite(value, name):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} harus berupa angka finite.")
        return value

    def aerodynamic_resistance(
        self,
        friction_velocity=None,
        reference_height=None,
        roughness_length=None
    ):
        u = self.friction_velocity if friction_velocity is None else friction_velocity
        zr = self.reference_height if reference_height is None else reference_height
        z0 = self.roughness_length if roughness_length is None else roughness_length

        u = self._finite(u, "Friction velocity")
        zr = self._finite(zr, "Reference height")
        z0 = self._finite(z0, "Roughness length")

        if u <= 0:
            raise ValueError("Friction velocity harus lebih besar dari 0.")

        if zr <= z0 or z0 <= 0:
            raise ValueError(
                "Reference height harus lebih besar dari roughness length positif."
            )

        return math.log(zr / z0) / (self.kappa * u)

    def surface_resistance(
        self,
        settling_velocity,
        friction_velocity=None,
        particle_diffusivity=1e-6
    ):
        if self.surface_resistance_value is not None:
            return {
                "schmidt": None,
                "stokes": None,
                "resistance": self.surface_resistance_value
            }

        u = self.friction_velocity if friction_velocity is None else friction_velocity
        u = self._finite(u, "Friction velocity")

        if u <= 0:
            raise ValueError("Friction velocity harus lebih besar dari 0.")

        nu = self.viscosity / self.rho
        diffusivity = max(float(particle_diffusivity), 1e-15)

        sc = nu / diffusivity

        st = (
            max(float(settling_velocity), 0.0) * u * u
            / max(self.g * nu, 1e-15)
        )

        rs = 1.0 / max(
            u * (
                sc ** (-2.0 / 3.0)
                + 10.0 ** (-3.0 / max(st, 1e-12))
            ),
            1e-15
        )

        return {
            "schmidt": sc,
            "stokes": st,
            "resistance": rs
        }

    def settling_velocity(
        self,
        radius,
        particle_density=2500.0,
        air_density=None
    ):
        r = self._finite(radius, "Particle radius")
        rho = self.rho if air_density is None else self._finite(
            air_density, "Air density"
        )

        d = 2.0 * r

        if r <= 0:
            raise ValueError("Particle radius harus positif.")

        delta = max(float(particle_density) - rho, 0.0)

        if delta <= 0:
            return 0.0

        v = delta * self.g * d * d / (18.0 * self.viscosity)
        re0 = rho * v * d / self.viscosity

        if re0 < 1.0:
            return float(v)

        for _ in range(200):
            re = rho * max(v, 1e-15) * d / self.viscosity

            cd = (
                24.0 / max(re, 1e-15)
                * (1.0 + 0.15 * re ** 0.687)
                if re < 1000 else 0.44
            )

            vn = math.sqrt(
                4.0 * d * self.g * delta
                / (3.0 * rho * max(cd, 1e-12))
            )

            if abs(vn - v) < max(1e-10, 1e-7 * vn):
                v = vn
                break

            v = 0.5 * v + 0.5 * vn

        return float(v)

    def dry_deposition_velocity(self, settling_velocity, ra, rs):
        vs = max(float(settling_velocity), 0.0)

        return (
            1.0 / max(
                float(ra) + float(rs)
                + float(ra) * float(rs) * vs,
                1e-15
            )
            + vs
        )

    def scavenging_coefficient(self, rain_rate, rh=None):
        rain = self._finite(
            0.0 if rain_rate is None else rain_rate,
            "Rain rate"
        )

        if rain < 0:
            raise ValueError("Rain rate tidak boleh negatif.")

        if rain == 0:
            return 0.0

        if rh is None:
            raise ValueError(
                "Relative humidity diperlukan untuk menentukan mekanisme wet scavenging."
            )

        humidity = self._finite(rh, "Relative humidity")

        if not 0.0 <= humidity <= 100.0:
            raise ValueError(
                "Relative humidity harus berada pada rentang 0-100%."
            )

        if humidity >= self.rh_threshold:
            a = self.rain_a_in_cloud
            b = self.rain_b_in_cloud
        else:
            a = self.rain_a_below_cloud
            b = self.rain_b_below_cloud

        return a * rain ** b

    def scavenging_type(self, rh):
        if rh is None:
            return None

        humidity = self._finite(rh, "Relative humidity")

        if not 0.0 <= humidity <= 100.0:
            raise ValueError(
                "Relative humidity harus berada pada rentang 0-100%."
            )

        return (
            "in-cloud"
            if humidity >= self.rh_threshold
            else "below-cloud"
        )

    @staticmethod
    def deposition_probability(rate_per_second, dt):
        rate = max(float(rate_per_second), 0.0)
        duration = max(float(dt), 0.0)

        return -math.expm1(-rate * duration)

    def wet_probability(self, rain_rate, dt, rh=None):
        duration = self._finite(dt, "Time step")

        if duration < 0:
            raise ValueError("Time step tidak boleh negatif.")

        coefficient_per_hour = self.scavenging_coefficient(
            rain_rate,
            rh
        )

        duration_hours = duration / 3600.0

        return -math.expm1(
            -coefficient_per_hour * duration_hours
        )

    def dry_probability(self, vd, layer_height=None, dt=300.0):
        height = (
            self.layer_height
            if layer_height is None
            else float(layer_height)
        )

        if height <= 0:
            raise ValueError("Layer height harus lebih besar dari 0.")

        return self.deposition_probability(
            max(float(vd), 0.0) / height,
            dt
        )

    def decide(
        self,
        particle,
        altitude,
        dt,
        friction_velocity=None,
        roughness_length=None,
        rain_rate=None,
        rh=None,
        rng=None
    ):
        rng = np.random.default_rng() if rng is None else rng

        altitude = self._finite(altitude, "Altitude")
        dt = self._finite(dt, "Time step")

        if dt < 0:
            raise ValueError("Time step tidak boleh negatif.")

        if altitude > self.layer_height:
            return {
                "deposited": False,
                "mechanism": None,
                "probability": 0.0,
                "pdry": 0.0,
                "pwet": 0.0,
                "scavenging_type": None
            }

        vs = self._finite(
            particle.get("settling_velocity", 0.0),
            "Settling velocity"
        )

        rain_rate = 0.0 if rain_rate is None else self._finite(
            rain_rate, "Rain rate"
        )

        if rain_rate < 0:
            raise ValueError("Rain rate tidak boleh negatif.")

        u_star = (
            self.friction_velocity
            if friction_velocity is None
            else self._finite(friction_velocity, "Friction velocity")
        )

        z0 = (
            self.roughness_length
            if roughness_length is None
            else self._finite(roughness_length, "Roughness length")
        )

        if u_star <= 0:
            raise ValueError("Friction velocity harus lebih besar dari 0.")

        if z0 <= 0 or z0 >= self.reference_height:
            raise ValueError(
                "Roughness length harus positif dan lebih kecil dari reference height."
            )

        ra = self.aerodynamic_resistance(
            friction_velocity=u_star,
            roughness_length=z0
        )

        rs = self.surface_resistance(
            vs,
            friction_velocity=u_star
        )["resistance"]

        vd = self.dry_deposition_velocity(vs, ra, rs)

        pdry = self.dry_probability(vd, dt=dt)
        pwet = self.wet_probability(rain_rate, dt, rh)

        ptotal = 1.0 - (1.0 - pdry) * (1.0 - pwet)

        deposited = bool(rng.random() < ptotal)

        if deposited:
            wet_only = pwet * (1.0 - pdry)

            mechanism = (
                "wet"
                if rng.random() < wet_only / max(ptotal, 1e-15)
                else "dry"
            )
        else:
            mechanism = None

        return {
            "deposited": deposited,
            "mechanism": mechanism,
            "probability": ptotal,
            "vd": vd,
            "pdry": pdry,
            "pwet": pwet,
            "scavenging_type": self.scavenging_type(rh)
                if rain_rate > 0 else None,
            "scavenging_coefficient": self.scavenging_coefficient(
                rain_rate, rh
            ),
            "ra": ra,
            "rs": rs,
            "vs": vs,
            "friction_velocity": u_star,
            "roughness_length": z0
        }
