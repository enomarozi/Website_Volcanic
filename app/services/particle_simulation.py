
import math
import numpy as np

class ParticleSimulationService:
    def __init__(self, meteorology, turbulence=None, particle=None, seed=42):
        self.meteorology = meteorology
        self.turbulence = turbulence
        self.particle = particle
        self.rng = np.random.default_rng(seed)


    def velocity(self, latitude, longitude, altitude, time_index):
        lat = float(latitude)
        lon = float(longitude)
        alt = float(altitude)

        if not all(math.isfinite(x) for x in (lat, lon, alt)):
            raise ValueError("Invalid particle coordinates.")

        if not -90 <= lat <= 90:
            raise ValueError("Latitude outside valid range.")

        if isinstance(time_index, (float, np.floating)):
            wind = self.meteorology.wind_components_at_elapsed(
                lat, lon, max(0.0, alt), time_index
            )
        else:
            wind = self.meteorology.wind_components_at_altitude(
                lat, lon, max(0.0, alt), int(time_index)
            )

        result = {
            "u": float(wind["u"]),
            "v": float(wind["v"]),
            "w": float(wind["w"])
        }

        if not all(math.isfinite(x) for x in result.values()):
            raise ValueError("GFS wind components contain non-finite values.")

        return result

    @staticmethod
    def _derivative(position, wind, settling_velocity, turbulence_velocity):
        lat = float(position[0])
        u = wind["u"] + turbulence_velocity[0]
        v = wind["v"] + turbulence_velocity[1]
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) < 1e-8:
            raise ValueError("Longitude integration is unstable near the poles.")

        return np.array([
            v / 111320.0,
            u / (111320.0 * cos_lat),
            wind["w"] + turbulence_velocity[2] - settling_velocity
        ], dtype=float)

    def step(self,latitude,longitude,altitude,dt,time_index, settling_velocity,turbulence_velocity=None):
        dt=float(dt)
        settling_velocity=float(settling_velocity)
        pos=np.array([latitude,longitude,altitude],dtype=float)
        turb=np.zeros(3) if turbulence_velocity is None else np.asarray(turbulence_velocity,dtype=float)

        if not math.isfinite(dt) or dt<=0:
            raise ValueError("Timestep must be positive.")
        if not math.isfinite(settling_velocity) or settling_velocity<0:
            raise ValueError("Settling velocity must be non-negative.")
        if turb.shape!=(3,) or not np.all(np.isfinite(turb)):
            raise ValueError("Invalid turbulence velocity.")
        if not np.all(np.isfinite(pos)):
            raise ValueError("Invalid particle position.")

        continuous=isinstance(time_index,(float,np.floating))
        t0=float(time_index)

        def derivative(position, offset):
            t = t0 + offset if continuous else time_index

            sample_position = position.copy()
            sample_position[2] = max(0.0, sample_position[2])

            wind = self.velocity(
                sample_position[0],
                sample_position[1],
                sample_position[2],
                t
            )

            return self._derivative(
                sample_position,
                wind,
                settling_velocity,
                turb
            ), wind

        try:
            k1,w1=derivative(pos,0.0)
            k2,_=derivative(pos+0.5*dt*k1,0.5*dt)
            k3,_=derivative(pos+0.5*dt*k2,0.5*dt)
            k4,_=derivative(pos+dt*k3,dt)
        except ValueError as exc:
            raise ValueError(
                f"RK4 error: {exc} | "
                f"initial_altitude={pos[2]:.2f} m | "
                f"dt={dt:.2f} s | "
                f"settling={settling_velocity:.4f} m/s | "
                f"turbulence_vertical={turb[2]:.4f} m/s | "
                f"time={t0}"
            ) from exc

        out=pos+(dt/6.0)*(k1+2*k2+2*k3+k4)
        if not np.all(np.isfinite(out)) or not -90<=out[0]<=90:
            raise ValueError("RK4 produced invalid particle coordinates.")

        out[1]=((out[1]+180.0)%360.0)-180.0

        return {
            "latitude":float(out[0]),
            "longitude":float(out[1]),
            "altitude":float(out[2]),
            "u":w1["u"]+turb[0],
            "v":w1["v"]+turb[1],
            "vertical":w1["w"]+turb[2]-settling_velocity,
            "time_index":self.meteorology.time_index_at_elapsed(t0) if continuous else int(time_index)
        }
