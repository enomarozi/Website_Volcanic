from pathlib import Path
import numpy as np
from netCDF4 import Dataset


class SurfaceMeteorologyService:

    def __init__(self, instant_file: str, accum_file: str):
        self.instant_file = Path(instant_file)
        self.accum_file = Path(accum_file)

    def load(self):
        if not self.instant_file.exists():
            raise FileNotFoundError(f"Instant NetCDF file not found: {self.instant_file}")

        if not self.accum_file.exists():
            raise FileNotFoundError(f"Accum NetCDF file not found: {self.accum_file}")

        with Dataset(self.instant_file) as instant, Dataset(self.accum_file) as accum:
            return {
                "time": np.asarray(instant.variables["valid_time"][:]),
                "latitude": np.asarray(instant.variables["latitude"][:]),
                "longitude": np.asarray(instant.variables["longitude"][:]),
                "friction_velocity": np.asarray(instant.variables["zust"][:], dtype=float),
                "temperature": np.asarray(instant.variables["t2m"][:], dtype=float),
                "heat_flux_accumulated": np.asarray(accum.variables["sshf"][:], dtype=float)
            }

    def accumulated_flux_to_wm2(self, value, interval_seconds=21600):
        return float(value) / float(interval_seconds)

    def surface_profile(self, latitude: float, longitude: float, time_index: int = 0):
        data = self.load()

        latitudes = data["latitude"]
        longitudes = data["longitude"]

        lat_index = np.abs(latitudes - latitude).argmin()
        lon_index = np.abs(longitudes - longitude).argmin()

        sshf = data["heat_flux_accumulated"][time_index, lat_index, lon_index]

        heat_flux_wm2 = self.accumulated_flux_to_wm2(sshf)

        return {
            "latitude": float(latitudes[lat_index]),
            "longitude": float(longitudes[lon_index]),
            "time_index": time_index,
            "friction_velocity": float(
                data["friction_velocity"][time_index, lat_index, lon_index]
            ),
            "temperature": float(
                data["temperature"][time_index, lat_index, lon_index]
            ),
            "heat_flux_accumulated": float(sshf),
            "heat_flux_wm2": float(heat_flux_wm2)
        }