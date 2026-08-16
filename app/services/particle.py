import numpy as np


class ParticleService:

    def __init__(
        self,
        particle_density: float = 2500.0,
        air_density: float = 1.225,
        gravity: float = 9.81,
        total_particles: int = 50
    ):
        self.particle_density = particle_density
        self.air_density = air_density
        self.gravity = gravity
        self.total_particles = total_particles

    def settling_velocity(self, radius):
        diameter = 2.0 * radius

        return (
            (self.particle_density - self.air_density)
            * self.gravity
            * diameter ** 2
        ) / (18.0 * 1.81e-5)

    def particle_mass(self, radius: float):
        if radius <= 0:
            raise ValueError("Particle radius must be greater than zero.")

        volume = (4.0 / 3.0) * np.pi * radius ** 3
        mass = self.particle_density * volume

        return float(mass)

    def particle_count(self):
        fine = int(self.total_particles * 0.16)
        medium = int(self.total_particles * 0.68)
        coarse = self.total_particles - fine - medium

        return {
            "fine": fine,
            "medium": medium,
            "coarse": coarse
        }

    def create_particles(
        self,
        total_mass,
        latitude,
        longitude,
        altitude
    ):
        if total_mass <= 0:
            raise ValueError("Total mass must be greater than zero.")

        counts = self.particle_count()

        definitions = [
            ("fine", 5e-6, 0.16, counts["fine"]),
            ("medium", 50e-6, 0.68, counts["medium"]),
            ("coarse", 500e-6, 0.16, counts["coarse"])
        ]

        particles = []
        particle_id = 0

        for name, radius, fraction, count in definitions:
            group_mass = total_mass * fraction
            mass_per_particle = group_mass / count
            settling_velocity = self.settling_velocity(radius)

            for _ in range(count):
                particles.append({
                    "id": particle_id,
                    "class": name,
                    "radius": float(radius),
                    "mass": float(mass_per_particle),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "altitude": float(altitude),
                    "settling_velocity": float(settling_velocity)
                })

                particle_id += 1

        return particles

    def create_particle_summary(self, total_mass: float):
        counts = self.particle_count()

        groups = []

        definitions = [
            ("fine", 5e-6, 0.16, counts["fine"]),
            ("medium", 50e-6, 0.68, counts["medium"]),
            ("coarse", 500e-6, 0.16, counts["coarse"])
        ]

        for name, radius, fraction, count in definitions:

            group_mass = total_mass * fraction

            mass_per_particle = group_mass / count

            groups.append({
                "class": name,
                "radius": float(radius),
                "fraction": float(fraction),
                "count": int(count),
                "mass": float(group_mass),
                "mass_per_particle": float(mass_per_particle)
            })

        return {
            "total_particles": self.total_particles,
            "total_mass": float(total_mass),
            "groups": groups
        }