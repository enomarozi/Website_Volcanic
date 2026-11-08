import math
import threading
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform


class LandCoverService:
    SOURCE = "ESA WorldCover 2021 v200"
    OCEAN_TILES = {
        "S09E102",
    }
    OCEAN_ROUGHNESS = 0.001
    ROUGHNESS = {
        10: 1.16,
        20: 0.16,
        30: 0.14,
        40: 0.13,
        50: 0.80,
        60: 0.01,
        70: 0.0024,
        80: 0.001,
        90: 0.56,
        95: 0.56,
        100: 0.0024,
    }

    LANDCOVER_NAMES = {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare/sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Direktori land cover tidak ditemukan: {self.data_dir}"
            )
        self._lock = threading.RLock()
        self._dataset_cache = {}
    def _ocean_surface(self, lat, lon, tile):
        return {
            "latitude": lat,
            "longitude": lon,
            "landcover_class": 80,
            "landcover_name": "Open ocean (fallback)",
            "roughness_length": self.OCEAN_ROUGHNESS,
            "roughness_units": "m",
            "landcover_source": "Ocean fallback",
            "landcover_tile": tile,
            "surface_type": "ocean"
        }
    def _validate_coordinates(self, lat, lon):
        lat = float(lat)
        lon = float(lon)

        if not (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):
            raise ValueError(f"Koordinat tidak valid: {lat}, {lon}")

        return lat, lon

    def _tile_name(self, lat, lon):
        lat_base = math.floor(lat / 3) * 3
        lon_base = math.floor(lon / 3) * 3

        lat_name = f"{'N' if lat_base >= 0 else 'S'}{abs(lat_base):02d}"
        lon_name = f"{'E' if lon_base >= 0 else 'W'}{abs(lon_base):03d}"

        return f"{lat_name}{lon_name}"

    def _get_tile(self, lat, lon):
        tile = self._tile_name(lat, lon)
        filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        path = self.data_dir / filename

        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Tile ESA WorldCover {tile} tidak ditemukan: {path}"
            )

        return path

    def _get_dataset(self, lat, lon):
        path = self._get_tile(lat, lon)
        key = str(path)

        with self._lock:
            if key not in self._dataset_cache:
                dataset = rasterio.open(path)

                if dataset.crs is None or dataset.count < 1:
                    dataset.close()
                    raise ValueError(f"File TIFF tidak valid: {path}")

                self._dataset_cache[key] = dataset

            return self._dataset_cache[key]

    def surface_at(self, lat, lon):
        lat, lon = self._validate_coordinates(lat, lon)
        tile = self._tile_name(lat, lon)
        path = self.data_dir / (
            f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        )

        if not path.is_file() and tile in self.OCEAN_TILES:
            return self._ocean_surface(lat, lon, tile)

        src = self._get_dataset(lat, lon)

        if src.crs.to_epsg() == 4326:
            x, y = lon, lat
        else:
            xs, ys = transform(
                "EPSG:4326",
                src.crs,
                [lon],
                [lat]
            )
            x, y = xs[0], ys[0]

        if not (
            src.bounds.left <= x < src.bounds.right
            and src.bounds.bottom < y <= src.bounds.top
        ):
            raise ValueError(
                f"Koordinat ({lat}, {lon}) di luar cakupan tile "
                f"{Path(src.name).name}"
            )

        with self._lock:
            values = next(src.sample([(x, y)]))

        value = values[0]

        if src.nodata is not None and value == src.nodata:
            raise ValueError(
                f"Data land cover NoData pada koordinat ({lat}, {lon}). "
                "Koordinat mungkin berada di laut atau area tanpa data."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"Data land cover tidak valid pada koordinat ({lat}, {lon})."
            )

        landcover_class = int(value)

        if landcover_class not in self.ROUGHNESS:
            raise ValueError(
                f"Kelas land cover {landcover_class} belum memiliki "
                "nilai roughness."
            )

        return {
            "latitude": lat,
            "longitude": lon,
            "landcover_class": landcover_class,
            "landcover_name": self.LANDCOVER_NAMES[landcover_class],
            "roughness_length": self.ROUGHNESS[landcover_class],
            "roughness_units": "m",
            "landcover_source": self.SOURCE,
            "landcover_tile": Path(src.name).name,
        }

    def close(self):
        with self._lock:
            for dataset in self._dataset_cache.values():
                dataset.close()

            self._dataset_cache.clear()