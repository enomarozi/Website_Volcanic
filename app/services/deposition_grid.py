import math
import numpy as np

class DepositionGridService:
    def __init__(self, cell_size=0.01):
        self.cell_size = float(cell_size)
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be greater than zero.")

    def _grid_index(self, latitude, longitude, min_latitude, min_longitude):
        row = int(math.floor((float(latitude) - min_latitude) / self.cell_size))
        col = int(math.floor((float(longitude) - min_longitude) / self.cell_size))
        return row, col

    def _cell_center(self, row, col, min_latitude, min_longitude):
        latitude = min_latitude + (row + 0.5) * self.cell_size
        longitude = min_longitude + (col + 0.5) * self.cell_size
        return latitude, longitude

    def _cell_area(self, latitude):
        radius = 6371000.0
        lat1 = math.radians(float(latitude) - self.cell_size / 2.0)
        lat2 = math.radians(float(latitude) + self.cell_size / 2.0)
        lon_width = math.radians(self.cell_size)
        return abs(radius ** 2 * (math.sin(lat2) - math.sin(lat1)) * lon_width)

    def accumulate(self, deposition_records):
        if deposition_records is None:
            deposition_records = []
        records = list(deposition_records)
        if not records:
            return {"cell_size": self.cell_size, "min_latitude": None, "max_latitude": None, "min_longitude": None, "max_longitude": None, "rows": 0, "cols": 0, "total_deposited_mass": 0.0, "cells": [], "geojson": {"type": "FeatureCollection", "features": []}}

        latitudes = np.asarray([float(record["latitude"]) for record in records], dtype=float)
        longitudes = np.asarray([float(record["longitude"]) for record in records], dtype=float)
        masses = np.asarray([max(float(record["mass"]), 0.0) for record in records], dtype=float)

        min_latitude = math.floor(float(np.min(latitudes)) / self.cell_size) * self.cell_size
        max_latitude = math.ceil(float(np.max(latitudes)) / self.cell_size) * self.cell_size
        min_longitude = math.floor(float(np.min(longitudes)) / self.cell_size) * self.cell_size
        max_longitude = math.ceil(float(np.max(longitudes)) / self.cell_size) * self.cell_size

        cells = {}
        for record, latitude, longitude, mass in zip(records, latitudes, longitudes, masses):
            row, col = self._grid_index(latitude, longitude, min_latitude, min_longitude)
            key = (row, col)
            if key not in cells:
                cells[key] = {"row": row, "col": col, "mass": 0.0, "particle_count": 0, "min_altitude": float(record.get("altitude", 0.0)), "max_altitude": float(record.get("altitude", 0.0))}
            cells[key]["mass"] += float(mass)
            cells[key]["particle_count"] += 1
            altitude = float(record.get("altitude", 0.0))
            cells[key]["min_altitude"] = min(cells[key]["min_altitude"], altitude)
            cells[key]["max_altitude"] = max(cells[key]["max_altitude"], altitude)

        output_cells = []
        features = []

        for key in sorted(cells):
            cell = cells[key]
            latitude, longitude = self._cell_center(cell["row"], cell["col"], min_latitude, min_longitude)
            area = self._cell_area(latitude)
            mass = float(cell["mass"])
            density = mass / area if area > 0.0 else 0.0
            north = latitude + self.cell_size / 2.0
            south = latitude - self.cell_size / 2.0
            east = longitude + self.cell_size / 2.0
            west = longitude - self.cell_size / 2.0

            output_cell = {"row": int(cell["row"]), "col": int(cell["col"]), "latitude": latitude, "longitude": longitude, "mass": mass, "area": area, "deposition_density": density, "particle_count": int(cell["particle_count"]), "min_altitude": float(cell["min_altitude"]), "max_altitude": float(cell["max_altitude"])}
            output_cells.append(output_cell)

            features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}, "properties": {"row": int(cell["row"]), "col": int(cell["col"]), "latitude": latitude, "longitude": longitude, "mass": mass, "area": area, "deposition_density": density, "particle_count": int(cell["particle_count"]), "min_altitude": float(cell["min_altitude"]), "max_altitude": float(cell["max_altitude"])}})

        total_deposited_mass = float(sum(cell["mass"] for cell in output_cells))
        rows = max((cell["row"] for cell in output_cells), default=-1) + 1
        cols = max((cell["col"] for cell in output_cells), default=-1) + 1

        return {"cell_size": self.cell_size, "min_latitude": float(min_latitude), "max_latitude": float(max_latitude), "min_longitude": float(min_longitude), "max_longitude": float(max_longitude), "rows": int(rows), "cols": int(cols), "total_deposited_mass": total_deposited_mass, "cells": output_cells, "geojson": {"type": "FeatureCollection", "features": features}}

    def from_dispersion(self, dispersion_result):
        if not isinstance(dispersion_result, dict):
            raise TypeError("dispersion_result must be a dictionary.")
        return self.accumulate(dispersion_result.get("deposition_records", []))