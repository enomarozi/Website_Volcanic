
import math
import unittest

from app.services.concentration import ConcentrationService
from app.services.particle import ParticleService
from app.services.deposition import DepositionService


class TestPhysicsRegression(unittest.TestCase):

    def test_concentration_mass_volume(self):
        service = ConcentrationService(default_dz=500.0)
        area = service.cell_area(-8.411, 0.1, 0.1)
        volume = service.cell_volume(-8.411, 0.1, 0.1, 500.0)

        self.assertGreater(area, 0)
        self.assertAlmostEqual(volume, area * 500.0)

    def test_particle_settling(self):
        service = ParticleService(total_particles=500)

        fine = service.settling_velocity(5e-6)
        medium = service.settling_velocity(50e-6)
        coarse = service.settling_velocity(500e-6)

        self.assertTrue(0 < fine < medium < coarse)

    def test_dry_deposition_probability(self):
        service = DepositionService()
        probability = service.dry_probability(
            vd=0.1,
            layer_height=100.0,
            dt=71.25
        )

        expected = 1.0 - math.exp(-0.1 * 71.25 / 100.0)
        self.assertAlmostEqual(probability, expected, places=12)

    def test_wet_deposition_zero_rain(self):
        service = DepositionService()

        self.assertEqual(
            service.wet_probability(0.0, 71.25),
            0.0
        )

    def test_concentration_before_release(self):
        service = ConcentrationService()

        particles = [
            {
                "mass": 100.0,
                "release_time": 0.0,
                "status": "active",
                "trajectory": [
                    {
                        "time": 0.0,
                        "latitude": -8.411,
                        "longitude": 116.457,
                        "altitude": 5000.0
                    },
                    {
                        "time": 100.0,
                        "latitude": -8.410,
                        "longitude": 116.458,
                        "altitude": 4900.0
                    }
                ]
            },
            {
                "mass": 200.0,
                "release_time": 50.0,
                "status": "active",
                "trajectory": [
                    {
                        "time": 50.0,
                        "latitude": -8.411,
                        "longitude": 116.457,
                        "altitude": 5000.0
                    },
                    {
                        "time": 100.0,
                        "latitude": -8.410,
                        "longitude": 116.458,
                        "altitude": 4900.0
                    }
                ]
            }
        ]

        result = service.accumulate(particles)

        first = result["time_series"][0]
        last = result["time_series"][-1]

        first_mass = sum(c["mass"] for c in first["cells"])
        last_mass = sum(c["mass"] for c in last["cells"])

        self.assertAlmostEqual(first_mass, 100.0)
        self.assertAlmostEqual(last_mass, 300.0)

    def test_surface_deposition_conservation(self):
        service = ConcentrationService()

        particles = [
            {
                "mass": 100.0,
                "deposition": {
                    "latitude": -8.411,
                    "longitude": 116.457,
                    "mass": 100.0
                }
            },
            {
                "mass": 200.0,
                "deposition": {
                    "latitude": -8.412,
                    "longitude": 116.458,
                    "mass": 200.0
                }
            }
        ]

        result = service.surface_deposition(particles)

        self.assertTrue(
            math.isclose(
                result["total_deposited_mass"],
                300.0,
                rel_tol=1e-12
            )
        )


if __name__ == "__main__":
    unittest.main()
