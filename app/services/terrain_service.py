
import os
import math
import fcntl
import urllib.request
from pathlib import Path
from threading import RLock
from collections import OrderedDict

import rasterio
from rasterio.errors import RasterioError


class TerrainService:
    BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
    SOURCE = "Copernicus DEM GLO-30"

    def __init__(self, data_dir=None, max_open_datasets=16, timeout=120):
        base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir or base_dir / "data/terrain/Copernicus_GLO30").resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_file = self.data_dir / "tileList.txt"
        if not self.catalog_file.is_file():
            raise FileNotFoundError(f"Katalog DEM tidak ditemukan: {self.catalog_file}")
        self.available_tiles = {
            x.strip() for x in self.catalog_file.read_text().splitlines() if x.strip()
        }
        self.max_open_datasets = max(1, int(max_open_datasets))
        self.timeout = timeout
        self.datasets = OrderedDict()
        self.lock = RLock()

    @staticmethod
    def tile_name(lat, lon):
        lat = float(lat)
        lon = float(lon)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError("Koordinat harus berupa angka valid.")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("Koordinat latitude/longitude di luar batas.")
        lat_degree = math.floor(lat)
        lon_degree = math.floor(lon)
        ns = "N" if lat_degree >= 0 else "S"
        ew = "E" if lon_degree >= 0 else "W"
        return (
            f"Copernicus_DSM_COG_10_{ns}{abs(lat_degree):02d}_00_"
            f"{ew}{abs(lon_degree):03d}_00_DEM"
        )

    def tile_path(self, tile):
        return self.data_dir / tile / f"{tile}.tif"

    def _valid_raster(self, path):
        try:
            with rasterio.open(path) as src:
                return src.count >= 1 and src.width > 0 and src.height > 0 and src.crs is not None
        except (RasterioError, OSError):
            return False

    def _download_tile(self, tile):
        if tile not in self.available_tiles:
            raise FileNotFoundError(f"Tile {tile} tidak tersedia dalam katalog Copernicus.")

        target = self.tile_path(tile)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_suffix(".lock")

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)

            if target.is_file() and self._valid_raster(target):
                return target

            target.unlink(missing_ok=True)
            temp_path = target.with_suffix(".tif.part")
            url = f"{self.BASE_URL}/{tile}/{tile}.tif"

            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "Website-Volcanic/1.0"}
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    with open(temp_path, "wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())

                if not self._valid_raster(temp_path):
                    raise IOError(f"GeoTIFF hasil unduhan tidak valid: {tile}")

                os.replace(temp_path, target)
                return target

            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _get_dataset(self, tile):
        with self.lock:
            if tile in self.datasets:
                self.datasets.move_to_end(tile)
                return self.datasets[tile]

            path = self.tile_path(tile)
            if not path.is_file() or not self._valid_raster(path):
                path = self._download_tile(tile)

            try:
                dataset = rasterio.open(path)
            except RasterioError as exc:
                raise IOError(f"Gagal membuka DEM {tile}: {exc}") from exc

            self.datasets[tile] = dataset
            while len(self.datasets) > self.max_open_datasets:
                _, old_dataset = self.datasets.popitem(last=False)
                old_dataset.close()

            return dataset

    def elevation_at(self, lat, lon):
        lat, lon = float(lat), float(lon)
        tile = self.tile_name(lat, lon)
        dataset = self._get_dataset(tile)

        if not (
            dataset.bounds.left <= lon <= dataset.bounds.right
            and dataset.bounds.bottom <= lat <= dataset.bounds.top
        ):
            raise ValueError(f"Koordinat berada di luar cakupan tile {tile}.")

        value = next(dataset.sample([(lon, lat)]))[0]
        if dataset.nodata is not None and value == dataset.nodata:
            raise ValueError(f"Nilai elevasi NoData pada koordinat {lat}, {lon}.")

        elevation = float(value)
        if not math.isfinite(elevation):
            raise ValueError("Nilai elevasi DEM tidak valid.")

        return {
            "elevation": elevation,
            "latitude": lat,
            "longitude": lon,
            "terrain_raster": Path(dataset.name).name,
            "terrain_tile": tile,
            "dem_source": self.SOURCE,
            "resolution": list(dataset.res),
            "crs": str(dataset.crs),
            "status": "available"
        }

    def get_tile_status(self, lat, lon):
        tile = self.tile_name(lat, lon)
        path = self.tile_path(tile)
        downloaded = path.is_file() and self._valid_raster(path)
        return {
            "tile": tile,
            "available_in_catalog": tile in self.available_tiles,
            "downloaded": downloaded,
            "file_path": str(path) if downloaded else None,
            "status": "cached" if downloaded else (
                "available" if tile in self.available_tiles else "unavailable"
            )
        }

    def close(self):
        with self.lock:
            for dataset in self.datasets.values():
                dataset.close()
            self.datasets.clear()
