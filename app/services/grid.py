
import math
import numpy as np

class EulerianGridService:
    EARTH_RADIUS = 6.371e6

    def __init__(self, resolution=1000.0, radius=100000.0, vertical_levels=None):
        self.resolution = float(resolution)
        self.radius = float(radius)
        self.vertical_levels = np.asarray(
            vertical_levels if vertical_levels is not None else
            [0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000],
            dtype=float
        )
        if self.resolution <= 0 or self.radius <= 0:
            raise ValueError("Resolution and radius must be greater than zero.")
        if self.vertical_levels.size == 0 or not np.all(np.isfinite(self.vertical_levels)):
            raise ValueError("Vertical levels must be finite and non-empty.")
        if np.any(np.diff(self.vertical_levels) <= 0):
            raise ValueError("Vertical levels must be strictly increasing.")

    def create_grid(self, latitude, longitude):
        latitude, longitude = float(latitude), float(longitude)
        cos_lat = max(abs(math.cos(math.radians(latitude))), 1e-8)
        lat_step = self.resolution / 111320.0
        lon_step = self.resolution / (111320.0 * cos_lat)
        lat_radius = self.radius / 111320.0
        lon_radius = self.radius / (111320.0 * cos_lat)
        return {
            "latitude": np.arange(latitude-lat_radius, latitude+lat_radius+lat_step*0.5, lat_step),
            "longitude": np.arange(longitude-lon_radius, longitude+lon_radius+lon_step*0.5, lon_step),
            "vertical": self.vertical_levels.copy()
        }

    @staticmethod
    def _index(values, value):
        value = float(value)
        if value < values[0] or value > values[-1]:
            return None
        return int(np.abs(values-value).argmin())

    def _cell_polygon(self, latitude, longitude):
        cos_lat = max(abs(math.cos(math.radians(latitude))), 1e-8)
        dlat = self.resolution / 111320.0
        dlon = self.resolution / (111320.0*cos_lat)
        south, north = latitude-dlat/2, latitude+dlat/2
        west, east = longitude-dlon/2, longitude+dlon/2
        return [[west,south],[east,south],[east,north],[west,north],[west,south]]

    def _layer_thickness(self, index):
        z = self.vertical_levels
        if len(z) == 1:
            return self.resolution
        if index == 0:
            lower = max(0.0, z[0]-(z[1]-z[0])/2)
        else:
            lower = (z[index-1]+z[index])/2
        if index == len(z)-1:
            upper = z[index]+(z[index]-z[index-1])/2
        else:
            upper = (z[index]+z[index+1])/2
        return max(upper-lower, 1e-9)

    def _cell_area(self, latitude):
        dphi = math.radians(self.resolution/111320.0)
        dlon = self.resolution / (111320.0*max(abs(math.cos(math.radians(latitude))), 1e-8))
        dlambda = math.radians(dlon)
        return self.EARTH_RADIUS**2 * max(math.cos(math.radians(latitude)), 0.0) * dphi * dlambda

    def accumulate(self, trajectories, source_latitude, source_longitude):
        if not trajectories:
            raise ValueError("Trajectories cannot be empty.")

        grid = self.create_grid(source_latitude, source_longitude)
        lats, lons, levels = grid["latitude"], grid["longitude"], grid["vertical"]
        area = self.resolution**2
        snapshots = {}
        deposition = np.zeros((len(lats), len(lons)), dtype=float)
        deposition_particles = np.zeros((len(lats), len(lons)), dtype=int)
        deposited_ids = set()
        all_times = sorted({
            float(point["time"])
            for particle in trajectories
            for point in particle.get("trajectory", [])
            if "time" in point
        })

        for particle in trajectories:
            dep = particle.get("deposition")
            if not dep:
                continue
            lat_idx = self._index(lats, dep["latitude"])
            lon_idx = self._index(lons, dep["longitude"])
            if lat_idx is None or lon_idx is None:
                continue
            mass = max(float(particle.get("mass", 0.0)), 0.0)
            deposition[lat_idx, lon_idx] += mass
            deposition_particles[lat_idx, lon_idx] += 1
            deposited_ids.add(particle.get("id"))

        for t in all_times:
            mass_grid = np.zeros((len(levels), len(lats), len(lons)), dtype=float)
            for particle in trajectories:
                if particle.get("status") == "deposited":
                    dep = particle.get("deposition") or {}
                    if dep.get("time") is not None and float(dep["time"]) <= t:
                        continue
                points = particle.get("trajectory", [])
                prior = [p for p in points if float(p.get("time", 0.0)) <= t]
                if not prior:
                    continue
                point = prior[-1]
                lat_idx = self._index(lats, point["latitude"])
                lon_idx = self._index(lons, point["longitude"])
                z_idx = self._index(levels, max(float(point.get("altitude", 0.0)), 0.0))
                if lat_idx is None or lon_idx is None or z_idx is None:
                    continue
                mass_grid[z_idx, lat_idx, lon_idx] += max(float(particle.get("mass", 0.0)), 0.0)

            concentration = np.zeros_like(mass_grid)
            for k, z in enumerate(levels):
                for i, lat in enumerate(lats):
                    volume = self._cell_area(lat) * self._layer_thickness(k)
                    concentration[k, i, :] = mass_grid[k, i, :] / max(volume, 1e-12)
            snapshots[t] = concentration

        if snapshots:
            last_concentration = snapshots[max(snapshots)]
            max_concentration = float(last_concentration.max())
        else:
            last_concentration = np.zeros((len(levels), len(lats), len(lons)))
            max_concentration = 0.0

        features = []
        for row, col in zip(*np.nonzero(deposition > 0)):
            lat, lon = float(lats[row]), float(lons[col])
            mass = float(deposition[row, col])
            cell_area = self._cell_area(lat)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [self._cell_polygon(lat, lon)]},
                "properties": {
                    "row": int(row), "col": int(col), "latitude": lat, "longitude": lon,
                    "mass": mass, "area": cell_area,
                    "deposition_density": mass/max(cell_area, 1e-12),
                    "particle_count": int(deposition_particles[row, col])
                }
            })

        return {
            "latitude": lats.tolist(),
            "longitude": lons.tolist(),
            "vertical": levels.tolist(),
            "resolution": self.resolution,
            "radius": self.radius,
            "concentration": last_concentration.tolist(),
            "concentration_time_series": [
                {"time": t, "concentration": snapshots[t].tolist()}
                for t in sorted(snapshots)
            ],
            "max_concentration": max_concentration,
            "max_deposition": float(deposition.max()),
            "total_grid_mass": float(deposition.sum()),
            "total_deposited_mass": float(deposition.sum()),
            "deposition": deposition.tolist(),
            "deposition_geojson": {"type": "FeatureCollection", "features": features}
        }
