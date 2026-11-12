import math
import numpy as np


class ParticleService:
    CLASSES = (
        ("fine", 5e-6, 0.16),
        ("medium", 50e-6, 0.68),
        ("coarse", 500e-6, 0.16)
    )

    def __init__(
        self,
        particle_density=2500.0,
        air_density=1.225,
        viscosity=1.81e-5,
        gravity=9.81,
        total_particles=50000,
        psd_mean_radius=50e-6,
        psd_sigma=0.7,
        seed=42,
        psd_method="discrete_mass"
    ):
        self.particle_density = float(particle_density)
        self.air_density = float(air_density)
        self.viscosity = float(viscosity)
        self.gravity = float(gravity)
        self.total_particles = int(total_particles)
        self.psd_mean_radius = float(psd_mean_radius)
        self.psd_sigma = float(psd_sigma)
        self.psd_method = str(psd_method)
        self.rng = np.random.default_rng(seed)
        self._settling_cache = {}

        values = (
            self.particle_density,
            self.air_density,
            self.viscosity,
            self.gravity,
            self.psd_mean_radius
        )

        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Invalid particle physical parameters.")

        if self.total_particles <= 0:
            raise ValueError("Total particles must be positive.")

        if not math.isfinite(self.psd_sigma) or self.psd_sigma < 0:
            raise ValueError("Invalid PSD sigma.")

        if self.psd_method not in ("discrete_mass", "lognormal_mass"):
            raise ValueError("Unsupported PSD method.")

    def settling_velocity(self, radius, air_density=None):
        r = float(radius)
        rho = self.air_density if air_density is None else float(air_density)

        if not math.isfinite(rho) or rho <= 0:
            raise ValueError("Invalid air density.")

        if not math.isfinite(r) or r <= 0:
            raise ValueError("Invalid particle radius.")

        key = (r, rho)

        if key in self._settling_cache:
            return self._settling_cache[key]

        d = 2.0 * r
        delta = self.particle_density - rho

        if delta <= 0:
            return 0.0

        v = delta * self.gravity * d**2 / (18.0 * self.viscosity)

        for _ in range(200):
            re = rho * v * d / self.viscosity

            if re < 1.0:
                break

            cd = (
                24.0 / re * (1.0 + 0.15 * re**0.687)
                if re < 1000.0 else 0.44
            )

            vn = math.sqrt(
                4.0 * d * self.gravity * delta /
                (3.0 * rho * cd)
            )

            if abs(vn - v) <= max(1e-10, 1e-7 * vn):
                v = vn
                break

            v = 0.5 * (v + vn)
        else:
            raise RuntimeError("Settling velocity did not converge.")

        self._settling_cache[key] = float(v)
        return float(v)

    def particle_mass(self, radius):
        r = float(radius)

        if not math.isfinite(r) or r <= 0:
            raise ValueError("Invalid radius.")

        return (
            self.particle_density *
            (4.0 / 3.0) *
            math.pi *
            r**3
        )

    @staticmethod
    def classify_radius(radius):
        r = float(radius)

        if not math.isfinite(r) or r <= 0:
            raise ValueError("Invalid radius.")

        if r <= 5e-6:
            return "fine"

        if r <= 50e-6:
            return "medium"

        return "coarse"

    def sample_radii(self, count):
        n = int(count)

        if n <= 0:
            raise ValueError("Invalid particle count.")

        if self.psd_method == "lognormal_mass":
            radii = self.rng.lognormal(
                mean=math.log(self.psd_mean_radius),
                sigma=self.psd_sigma,
                size=n
            )
            return radii

        fractions = np.array(
            [item[2] for item in self.CLASSES],
            dtype=float
        )

        counts = self.rng.multinomial(n, fractions)

        radii = np.concatenate([
            np.full(count, radius)
            for count, (_, radius, _) in zip(counts, self.CLASSES)
        ])

        self.rng.shuffle(radii)
        return radii

    def create_particles(
        self,
        total_mass,
        latitude,
        longitude,
        altitude,
        eruption_height=None,
        eruption_duration=None
    ):
        total_mass = float(total_mass)
        lat = float(latitude)
        lon = float(longitude)
        alt = float(altitude)

        if not math.isfinite(total_mass) or total_mass <= 0:
            raise ValueError("Invalid total mass.")

        if not all(math.isfinite(x) for x in (lat, lon, alt)):
            raise ValueError("Invalid coordinates.")

        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("Coordinates outside valid range.")

        radii = self.sample_radii(self.total_particles)
        if eruption_height is not None:
            height = float(eruption_height)

            if not math.isfinite(height) or height <= 0:
                raise ValueError("Invalid eruption height.")

            fractions = self.rng.beta(
                2.0, 2.0, self.total_particles
            )

            release_altitudes = alt + fractions * height
        else:
            release_altitudes = np.full(
                self.total_particles, alt
            )

        if eruption_duration is not None:
            release_duration = float(eruption_duration)

            if not math.isfinite(release_duration) or release_duration <= 0:
                raise ValueError("Invalid eruption duration.")

            release_times = self.rng.uniform(
                0.0, release_duration, self.total_particles
            )
        else:
            release_times = np.zeros(self.total_particles)

        mass_each = total_mass / self.total_particles

        unique_radii = np.unique(radii)

        settling = {
            float(r): self.settling_velocity(r)
            for r in unique_radii
        }

        classes = {
            float(r): self.classify_radius(r)
            for r in unique_radii
        }

        particles = []

        for i, radius in enumerate(radii):
            r = float(radius)

            particles.append({
                "id": i,
                "particle_id": i,
                "class": classes[r],
                "particle_class": classes[r],
                "radius": r,
                "mass": mass_each,
                "latitude": lat,
                "longitude": lon,
                "altitude": float(release_altitudes[i]),
                "release_time": float(release_times[i]),
                "settling_velocity": settling[r],
                "status": "active"
            })
        
        print("\n========== PARTICLE PSD VALIDATION ==========")

        total_particle_mass = math.fsum(
            p["mass"] for p in particles
        )

        print(f"Total particles    : {len(particles)}")
        print(f"Expected mass      : {total_mass:.6f} kg")
        print(f"Particle mass sum  : {total_particle_mass:.6f} kg")
        print(f"Mass difference    : {total_particle_mass - total_mass:.9f} kg")

        for name, radius, fraction in self.CLASSES:
            selected = [
                p for p in particles
                if p["class"] == name
            ]

            count = len(selected)
            class_mass = math.fsum(
                p["mass"] for p in selected
            )

            actual_fraction = (
                class_mass / total_particle_mass
                if total_particle_mass > 0 else 0.0
            )

            velocity = self.settling_velocity(radius)
            diameter = 2.0 * radius
            reynolds = (
                self.air_density * velocity * diameter
                / self.viscosity
            )

            print(f"\nClass             : {name}")
            print(f"Radius            : {radius:.9f} m")
            print(f"Target fraction   : {fraction:.6f}")
            print(f"Actual fraction   : {actual_fraction:.6f}")
            print(f"Particle count    : {count}")
            print(f"Class mass        : {class_mass:.6f} kg")
            print(f"Settling velocity : {velocity:.9f} m/s")
            print(f"Reynolds number   : {reynolds:.6f}")

        mass_valid = math.isclose(
            total_particle_mass,
            total_mass,
            rel_tol=1e-12,
            abs_tol=1e-6
        )

        print(f"\nMass conservation valid: {mass_valid}")
        print("============================================\n")

        return particles

    def create_particle_summary(self, total_mass):
        total_mass = float(total_mass)

        if not math.isfinite(total_mass) or total_mass <= 0:
            raise ValueError("Invalid total mass.")

        return {
            "total_particles": self.total_particles,
            "total_mass": total_mass,
            "mass_per_particle": total_mass / self.total_particles,
            "distribution": {
                "type": self.psd_method,
                "mean_radius_m": self.psd_mean_radius,
                "sigma": self.psd_sigma,
                "classes": [
                    {
                        "name": name,
                        "radius_m": radius,
                        "fraction": fraction,
                        "count": round(fraction * self.total_particles)
                    }
                    for name, radius, fraction in self.CLASSES
                ] if self.psd_method == "discrete_mass" else []
            }
        }