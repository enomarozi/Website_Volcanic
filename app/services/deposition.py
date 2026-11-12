
import math
import numpy as np

class DepositionService:
    def __init__(
        self, gravity=9.81, von_karman=0.4, air_density=1.225,
        viscosity=1.81e-5, rain_a=8.4e-5, rain_b=0.79,
        rain_a_below_cloud=3.36e-4, rain_b_below_cloud=0.305,
        rh_threshold=80.0, reference_height=10.0,
        roughness_length=0.0133, friction_velocity=0.272,
        layer_height=100.0, surface_resistance_value=2.53,
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

        positive = (
            self.g, self.kappa, self.rho, self.viscosity,
            self.reference_height, self.layer_height,
            self.roughness_length, self.friction_velocity
        )
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Parameter fisika harus finite dan positif.")
        if self.reference_height <= self.roughness_length:
            raise ValueError("Reference height harus lebih besar dari roughness length.")
        if not 0 <= self.rh_threshold <= 100:
            raise ValueError("RH threshold harus 0-100%.")
        if not all(math.isfinite(x) and x >= 0 for x in (
            self.rain_a_in_cloud, self.rain_b_in_cloud,
            self.rain_a_below_cloud, self.rain_b_below_cloud,
            self.default_rain_rate
        )):
            raise ValueError("Parameter hujan harus finite dan non-negatif.")
        if self.surface_resistance_value is not None:
            if not math.isfinite(self.surface_resistance_value) or self.surface_resistance_value < 0:
                raise ValueError("Surface resistance harus finite dan non-negatif.")

    @staticmethod
    def _finite(value, name):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} harus finite.")
        return value

    def aerodynamic_resistance(self, friction_velocity=None,
                               reference_height=None, roughness_length=None):
        u = self._finite(
            self.friction_velocity if friction_velocity is None else friction_velocity,
            "Friction velocity"
        )
        zr = self._finite(
            self.reference_height if reference_height is None else reference_height,
            "Reference height"
        )
        z0 = self._finite(
            self.roughness_length if roughness_length is None else roughness_length,
            "Roughness length"
        )
        if u <= 0 or z0 <= 0 or zr <= z0:
            raise ValueError("Diperlukan u*>0 dan reference_height>z0>0.")
        return math.log(zr / z0) / (self.kappa * u)

    def surface_resistance(self, settling_velocity,
                           friction_velocity=None, particle_diffusivity=1e-6):
        if self.surface_resistance_value is not None:
            return {
                "schmidt": None,
                "stokes": None,
                "resistance": self.surface_resistance_value
            }

        u = self._finite(
            self.friction_velocity if friction_velocity is None else friction_velocity,
            "Friction velocity"
        )
        vs = self._finite(settling_velocity, "Settling velocity")
        diffusivity = self._finite(particle_diffusivity, "Particle diffusivity")
        if u <= 0 or vs < 0 or diffusivity <= 0:
            raise ValueError("Parameter surface resistance tidak valid.")

        nu = self.viscosity / self.rho
        sc = nu / diffusivity
        st = vs * u * u / (self.g * nu)
        efficiency = sc ** (-2.0 / 3.0) + 10.0 ** (-3.0 / max(st, 1e-12))
        rs = 1.0 / max(u * efficiency, 1e-15)
        return {"schmidt": sc, "stokes": st, "resistance": rs}

    def settling_velocity(self, radius, particle_density=2500.0, air_density=None):
        r = self._finite(radius, "Particle radius")
        rho = self._finite(
            self.rho if air_density is None else air_density, "Air density"
        )
        rho_p = self._finite(particle_density, "Particle density")
        if r <= 0 or rho <= 0 or rho_p <= 0:
            raise ValueError("Radius dan densitas harus positif.")

        delta = max(rho_p - rho, 0.0)
        if delta == 0:
            return 0.0

        d = 2.0 * r
        v = delta * self.g * d * d / (18.0 * self.viscosity)
        if rho * v * d / self.viscosity < 1.0:
            return float(v)

        for _ in range(200):
            re = rho * max(v, 1e-15) * d / self.viscosity
            cd = (
                24.0 / max(re, 1e-15) * (1.0 + 0.15 * re ** 0.687)
                if re < 1000.0 else 0.44
            )
            vn = math.sqrt(4.0 * d * self.g * delta / (3.0 * rho * cd))
            if abs(vn - v) < max(1e-10, 1e-7 * vn):
                v = vn
                break
            v = 0.5 * (v + vn)
        return float(v)

    def dry_deposition_velocity(self, settling_velocity, ra, rs):
        vs = self._finite(settling_velocity, "Settling velocity")
        ra = self._finite(ra, "Aerodynamic resistance")
        rs = self._finite(rs, "Surface resistance")
        if vs < 0 or ra < 0 or rs < 0:
            raise ValueError("Parameter dry deposition tidak boleh negatif.")
        return vs + 1.0 / max(ra + rs + ra * rs * vs, 1e-15)

    def scavenging_type(self, rh):
        if rh is None:
            return None
        humidity = self._finite(rh, "Relative humidity")
        if not 0 <= humidity <= 100:
            raise ValueError("Relative humidity harus 0-100%.")
        return "in-cloud" if humidity >= self.rh_threshold else "below-cloud"

    def scavenging_coefficient(self, rain_rate, rh=None):
        rain = self._finite(
            0.0 if rain_rate is None else rain_rate, "Rain rate"
        )
        if rain < 0:
            raise ValueError("Rain rate tidak boleh negatif.")
        if rain == 0:
            return 0.0
        if rh is None:
            raise ValueError("Relative humidity diperlukan ketika hujan > 0.")

        kind = self.scavenging_type(rh)
        if kind == "in-cloud":
            return self.rain_a_in_cloud * rain ** self.rain_b_in_cloud
        return self.rain_a_below_cloud * rain ** self.rain_b_below_cloud

    @staticmethod
    def deposition_probability(rate_per_second, dt):
        rate = float(rate_per_second)
        duration = float(dt)
        if not math.isfinite(rate) or not math.isfinite(duration):
            raise ValueError("Rate dan timestep harus finite.")
        if rate < 0 or duration < 0:
            raise ValueError("Rate dan timestep tidak boleh negatif.")
        return -math.expm1(-rate * duration)

    def wet_probability(self, rain_rate, dt, rh=None):
        coefficient_per_second = self.scavenging_coefficient(rain_rate, rh)
        return self.deposition_probability(coefficient_per_second, dt)

    def dry_probability(self, vd, layer_height=None, dt=300.0):
        height = self._finite(
            self.layer_height if layer_height is None else layer_height,
            "Layer height"
        )
        velocity = self._finite(vd, "Dry deposition velocity")
        if height <= 0 or velocity < 0:
            raise ValueError("Layer height harus positif dan vd non-negatif.")
        return self.deposition_probability(velocity / height, dt)

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

        altitude = self._finite(altitude, "Altitude AGL")
        dt = self._finite(dt, "Time step")

        if dt < 0:
            raise ValueError("Time step tidak boleh negatif.")

        vs = self._finite(
            particle.get("settling_velocity", 0.0),
            "Settling velocity"
        )

        rain = self._finite(
            0.0 if rain_rate is None else rain_rate,
            "Rain rate"
        )

        if vs < 0 or rain < 0:
            raise ValueError("Parameter deposisi tidak valid.")

        near_surface = 0.0 <= altitude <= self.layer_height

        dry_rate = 0.0
        wet_rate = self.scavenging_coefficient(rain, rh)

        vd = 0.0
        ra = None
        rs = None

        u_star = None
        z0 = None

        if near_surface:
            u_star = self._finite(
                self.friction_velocity
                if friction_velocity is None
                else friction_velocity,
                "Friction velocity"
            )

            z0 = self._finite(
                self.roughness_length
                if roughness_length is None
                else roughness_length,
                "Roughness length"
            )

            if u_star <= 0:
                raise ValueError("Friction velocity harus positif.")

            if z0 <= 0 or z0 >= self.reference_height:
                raise ValueError("Roughness length tidak valid.")

            ra = self.aerodynamic_resistance(
                friction_velocity=u_star,
                roughness_length=z0
            )

            rs = self.surface_resistance(
                vs,
                friction_velocity=u_star
            )["resistance"]

            vd = self.dry_deposition_velocity(vs, ra, rs)
            dry_rate = vd / self.layer_height

        total_rate = dry_rate + wet_rate

        pdry = self.deposition_probability(dry_rate, dt)
        pwet = self.deposition_probability(wet_rate, dt)
        ptotal = self.deposition_probability(total_rate, dt)

        deposited = bool(rng.random() < ptotal)
        mechanism = None

        if deposited and total_rate > 0:
            mechanism = (
                "wet"
                if rng.random() < wet_rate / total_rate
                else "dry"
            )

        return {
            "deposited": deposited,
            "mechanism": mechanism,
            "probability": ptotal,
            "vd": vd,
            "pdry": pdry,
            "pwet": pwet,
            "scavenging_type": (
                self.scavenging_type(rh)
                if rain > 0 else None
            ),
            "scavenging_coefficient": wet_rate,
            "ra": ra,
            "rs": rs,
            "vs": vs,
            "friction_velocity": u_star,
            "roughness_length": z0
        }
