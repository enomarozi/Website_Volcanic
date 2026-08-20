import numpy as np


class ParticleService:

    def __init__(
        self,
        particle_density=2500.0,
        air_density=1.225,
        gravity=9.81,
        total_particles=500
    ):
        self.particle_density = particle_density
        self.air_density = air_density
        self.gravity = gravity
        self.total_particles = total_particles

    def settling_velocity(self, radius):
        d = 2.0 * float(radius)
        mu = 1.81e-5

        return float(
            (
                (self.particle_density - self.air_density)
                * self.gravity
                * d ** 2
            ) / (18.0 * mu)
        )

    def particle_mass(self, radius):
        if radius <= 0:
            raise ValueError("Radius must be greater than zero.")

        return float(
            self.particle_density
            * (4.0 / 3.0)
            * np.pi
            * radius ** 3
        )

    def particle_count(self):
        fine = int(self.total_particles * 0.16)
        medium = int(self.total_particles * 0.68)
        coarse = self.total_particles - fine - medium

        return {
            "fine": fine,
            "medium": medium,
            "coarse": coarse
        }

    def definitions(self):
        counts = self.particle_count()

        return [
            ("fine", 5e-6, 0.16, counts["fine"]),
            ("medium", 50e-6, 0.68, counts["medium"]),
            ("coarse", 500e-6, 0.16, counts["coarse"])
        ]

    def create_particles(
        self,
        total_mass,
        latitude,
        longitude,
        altitude
    ):
        if total_mass <= 0:
            raise ValueError("Total mass must be greater than zero.")

        particles = []
        particle_id = 0

        for name, radius, fraction, count in self.definitions():
            if count <= 0:
                continue

            group_mass = total_mass * fraction
            mass_each = group_mass / count
            settling = self.settling_velocity(radius)

            for _ in range(count):
                particles.append({
                    "id": particle_id,
                    "class": name,
                    "radius": radius,
                    "fraction": fraction,
                    "mass": mass_each,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "altitude": float(altitude),
                    "settling_velocity": settling
                })
                particle_id += 1

        return particles

    def create_particle_summary(self, total_mass):
        groups = []

        for name, radius, fraction, count in self.definitions():
            group_mass = total_mass * fraction

            groups.append({
                "class": name,
                "radius": radius,
                "fraction": fraction,
                "count": count,
                "mass": group_mass,
                "mass_per_particle":
                    group_mass / count if count else 0.0,
                "settling_velocity":
                    self.settling_velocity(radius)
            })

        return {
            "total_particles": self.total_particles,
            "total_mass": float(total_mass),
            "groups": groups
        }