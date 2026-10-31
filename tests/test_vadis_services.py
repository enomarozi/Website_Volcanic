
import math
import numpy as np
from app.services.eruption import EruptionService
from app.services.particle import ParticleService
from app.services.particle_simulation import ParticleSimulationService
from app.services.turbulence import TurbulenceService
from app.services.deposition import DepositionService
from app.services.dispersion import DispersionService
from app.services.concentration import ConcentrationService
from app.services.geojson import GeoJSONService

def test_mer_positive():
    profile = [
        {"altitude": 0, "u": 5, "v": 2, "potential_temperature": 300},
        {"altitude": 5000, "u": 10, "v": 3, "potential_temperature": 320},
        {"altitude": 10000, "u": 15, "v": 4, "potential_temperature": 340},
    ]
    result = EruptionService().calculate(profile, 10000, 3600)
    assert result["mer"] > 0
    assert math.isclose(result["total_mass"], result["mer"] * 3600)
    assert result["brunt_vaisala"] > 0

def test_particle_mass_conservation():
    service = ParticleService(total_particles=100, seed=42)
    particles = service.create_particles(1e9, -7.54, 110.44, 5000)
    assert len(particles) == 100
    assert all(p["mass"] > 0 for p in particles)
    assert math.isclose(sum(p["mass"] for p in particles), 1e9, rel_tol=1e-12)
    assert all(p["settling_velocity"] >= 0 for p in particles)

def test_turbulence_finite():
    service = TurbulenceService()
    rng = np.random.default_rng(42)
    value = service.langevin_velocity(0, 1.0, 100.0, 10.0, rng)
    assert math.isfinite(value)

def test_deposition_probability_bounds():
    service = DepositionService()
    p = service.deposition_probability(0.001, 60)
    assert 0 <= p <= 1
    assert service.deposition_probability(0, 60) == 0

def test_concentration_mass_and_geojson():
    particles = [{
        "particle_id": 1, "particle_class": "fine", "mass": 10.0,
        "status": "active", "trajectory": [
            {"time": 0, "latitude": -7.5, "longitude": 110.4, "altitude": 1000},
            {"time": 60, "latitude": -7.49, "longitude": 110.41, "altitude": 900},
        ]
    }]
    concentration = ConcentrationService().accumulate(particles)
    assert concentration["max_concentration"] > 0
    assert concentration["cell_count"] > 0
    geojson = GeoJSONService().trajectory_to_feature_collection(
        {"trajectories": particles}
    )
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["geometry"]["type"] == "LineString"
