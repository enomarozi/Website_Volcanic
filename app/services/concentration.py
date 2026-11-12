
import math
from bisect import bisect_right


class ConcentrationService:
    def __init__(self, earth_radius=6.371e6, default_dz=500.0):
        self.earth_radius = float(earth_radius)
        self.default_dz = float(default_dz)

        if self.earth_radius <= 0 or self.default_dz <= 0:
            raise ValueError("Earth radius and vertical resolution must be positive.")

    def cell_area(self, latitude, dlat=0.1, dlon=0.1):
        latitude = float(latitude)
        dlat = float(dlat)
        dlon = float(dlon)

        if dlat <= 0 or dlon <= 0:
            raise ValueError("Grid resolution must be positive.")

        phi = math.radians(latitude)
        dphi = math.radians(dlat)
        dlambda = math.radians(dlon)

        return (
            self.earth_radius ** 2
            * abs(math.cos(phi))
            * dphi
            * dlambda
        )

    def cell_volume(self, latitude, dlat=0.1, dlon=0.1, dz=None):
        dz = self.default_dz if dz is None else float(dz)

        if dz <= 0:
            raise ValueError("Vertical grid size must be positive.")

        return self.cell_area(latitude, dlat, dlon) * dz

    def grid_index(
        self,
        latitude,
        longitude,
        altitude,
        origin_latitude,
        origin_longitude,
        dlat=0.1,
        dlon=0.1,
        dz=None
    ):
        dz = self.default_dz if dz is None else float(dz)

        if min(dlat, dlon, dz) <= 0:
            raise ValueError("Grid resolutions must be positive.")

        return (
            math.floor((float(latitude) - float(origin_latitude)) / dlat),
            math.floor((float(longitude) - float(origin_longitude)) / dlon),
            math.floor(max(float(altitude), 0.0) / dz)
        )

    def accumulate(
        self,
        trajectories,
        origin_latitude=None,
        origin_longitude=None,
        dlat=0.1,
        dlon=0.1,
        dz=None,
        active_only=True
    ):
        dlat = float(dlat)
        dlon = float(dlon)
        dz = self.default_dz if dz is None else float(dz)

        if not all(
            math.isfinite(x) and x > 0
            for x in (dlat, dlon, dz)
        ):
            raise ValueError("Grid resolutions must be finite and positive.")

        empty = {
            "time_series": [],
            "grid": {},
            "metadata": {},
            "max_concentration": 0.0,
            "cell_count": 0,
            "grid_size": dlat
        }

        if not trajectories:
            return empty

        valid = [
            p for p in trajectories
            if p.get("trajectory")
        ]

        if not valid:
            return empty

        if origin_latitude is None:
            origin_latitude = min(
                float(p["trajectory"][0]["latitude"])
                for p in valid
            )

        if origin_longitude is None:
            origin_longitude = min(
                float(p["trajectory"][0]["longitude"])
                for p in valid
            )

        origin_latitude = float(origin_latitude)
        origin_longitude = float(origin_longitude)

        prepared = []
        all_times = set()

        for particle in valid:
            states = sorted(
                particle["trajectory"],
                key=lambda s: float(s["time"])
            )

            state_times = [
                float(state["time"])
                for state in states
            ]

            if any(
                not math.isfinite(t)
                for t in state_times
            ):
                raise ValueError("Invalid trajectory time.")

            release_time = float(
                particle.get("release_time", state_times[0])
            )

            if not math.isfinite(release_time):
                raise ValueError("Invalid particle release time.")

            deposition = particle.get("deposition") or {}
            deposition_time = deposition.get("time")

            if deposition_time is not None:
                deposition_time = float(deposition_time)

            mass = float(particle.get("mass", 0.0))

            if not math.isfinite(mass) or mass < 0:
                raise ValueError("Invalid particle mass.")

            prepared.append({
                "states": states,
                "times": state_times,
                "release_time": release_time,
                "deposition_time": deposition_time,
                "status": particle.get("status"),
                "mass": mass
            })

            all_times.update(state_times)

        times = sorted(all_times)
        series = []
        final_grid = {}
        max_concentration = 0.0
        max_cells = 0

        for t in times:
            cells = {}

            for particle in prepared:
                if t + 1e-9 < particle["release_time"]:
                    continue

                deposition_time = particle["deposition_time"]

                if (
                    active_only
                    and particle["status"] == "deposited"
                    and deposition_time is not None
                    and deposition_time <= t + 1e-9
                ):
                    continue

                index = bisect_right(
                    particle["times"],
                    t + 1e-9
                ) - 1

                if index < 0:
                    continue

                state = particle["states"][index]

                key = self.grid_index(
                    state["latitude"],
                    state["longitude"],
                    state["altitude"],
                    origin_latitude,
                    origin_longitude,
                    dlat,
                    dlon,
                    dz
                )

                cells[key] = (
                    cells.get(key, 0.0)
                    + particle["mass"]
                )

            output = []
            snapshot = {}

            for (i, j, k), mass in cells.items():
                lat = origin_latitude + (i + 0.5) * dlat
                lon = origin_longitude + (j + 0.5) * dlon

                volume = self.cell_volume(
                    lat, dlat, dlon, dz
                )

                concentration = mass / volume

                item = {
                    "i": i,
                    "j": j,
                    "k": k,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": (k + 0.5) * dz,
                    "mass": mass,
                    "volume": volume,
                    "concentration": concentration
                }

                output.append(item)

                snapshot[str((i, j, k))] = {
                    "mass": mass,
                    "volume": volume,
                    "concentration": concentration
                }

                max_concentration = max(
                    max_concentration,
                    concentration
                )

            max_cells = max(max_cells, len(output))

            series.append({
                "time": t,
                "cells": output
            })

            final_grid = snapshot

        return {
            "time_series": series,
            "grid": final_grid,
            "max_concentration": max_concentration,
            "cell_count": len(final_grid),
            "grid_size": dlat,
            "metadata": {
                "origin_latitude": origin_latitude,
                "origin_longitude": origin_longitude,
                "dlat": dlat,
                "dlon": dlon,
                "dz": dz,
                "earth_radius": self.earth_radius,
                "units": "kg/m3",
                "grid_represents": "last available time snapshot",
                "max_cells_per_snapshot": max_cells
            }
        }

    def surface_deposition(
        self,
        deposited_particles,
        dlat=0.1,
        dlon=0.1,
        origin_latitude=0.0,
        origin_longitude=0.0
    ):
        dlat = float(dlat)
        dlon = float(dlon)

        if dlat <= 0 or dlon <= 0:
            raise ValueError("Grid resolutions must be positive.")

        cells = {}

        for particle in deposited_particles:
            deposition = particle.get("deposition")

            if not deposition:
                continue

            lat = float(deposition["latitude"])
            lon = float(deposition["longitude"])

            i = math.floor(
                (lat - float(origin_latitude)) / dlat
            )
            j = math.floor(
                (lon - float(origin_longitude)) / dlon
            )

            key = (i, j)

            if key not in cells:
                cells[key] = {
                    "i": i,
                    "j": j,
                    "mass": 0.0,
                    "latitude": float(origin_latitude) + (i + 0.5) * dlat,
                    "longitude": float(origin_longitude) + (j + 0.5) * dlon,
                    "particle_count": 0
                }

            mass = float(
                deposition.get(
                    "mass",
                    particle.get("mass", 0.0)
                )
            )

            cells[key]["mass"] += mass
            cells[key]["particle_count"] += 1

        for cell in cells.values():
            area = self.cell_area(
                cell["latitude"],
                dlat,
                dlon
            )

            cell["area"] = area
            cell["deposition"] = cell["mass"] / area

        values = list(cells.values())

        return {
            "cells": values,
            "total_deposited_mass": math.fsum(
                cell["mass"] for cell in values
            ),
            "cell_count": len(values),
            "units": "kg/m2",
            "dlat": dlat,
            "dlon": dlon
        }

    def plume_metrics(
        self,
        mean_wind_speed,
        duration,
        plume_width_km=20.0,
        dlat=0.1,
        dlon=0.1,
        latitude=0.0
    ):
        length_km = (
            max(float(mean_wind_speed), 0.0)
            * max(float(duration), 0.0)
            / 1000.0
        )

        footprint_km2 = (
            length_km
            * max(float(plume_width_km), 0.0)
        )

        cell_km2 = (
            self.cell_area(latitude, dlat, dlon)
            / 1e6
        )

        return {
            "plume_length_km": length_km,
            "plume_width_km": float(plume_width_km),
            "footprint_km2": footprint_km2,
            "cell_area_km2": cell_km2,
            "estimated_affected_cells": (
                footprint_km2 / cell_km2
                if cell_km2 else 0.0
            )
        }
