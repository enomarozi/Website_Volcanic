
from app.services.particle_simulation import ParticleSimulationService
from app.services.dispersion import DispersionService

class FakeMeteorology:
    def time_information(self):
        return {"elapsed_seconds": [0, 60, 120]}

    def time_index_at_elapsed(self, elapsed):
        return min(2, max(0, round(elapsed / 60)))

    def wind_at_altitude(self, latitude, longitude, altitude, time_index):
        return {"u": 10.0, "v": 0.0, "w": 0.0}

def test_dispersion_integration():
    meteo = FakeMeteorology()
    particle_sim = ParticleSimulationService(meteo, seed=42)
    dispersion = DispersionService(particle_sim, meteo)

    particles = [{
        "particle_id": 1,
        "mass": 10.0,
        "latitude": -7.5,
        "longitude": 110.4,
        "altitude": 1000.0,
        "settling_velocity": 0.0,
        "status": "active",
    }]

    result = dispersion.simulate(particles, duration=90, dt=60)

    assert result["particle_count"] == 1
    path = result["trajectories"][0]["trajectory"]
    assert len(path) == 3
    assert path[-1]["time"] == 90
    assert path[-1]["longitude"] > path[0]["longitude"]
    assert path[-1]["altitude"] >= 0
