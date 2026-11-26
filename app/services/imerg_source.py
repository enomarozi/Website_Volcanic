import hashlib
import math
import re
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import BytesIO
from uuid import uuid4

import earthaccess
import numpy as np
import xarray as xr


class IMERGSource:
    SHORT_NAME = "GPM_3IMERGHHL"
    VERSION = "07"
    RESOLUTION = 0.1
    INTERVAL = 1800

    def __init__(self, cache_dir, timeout=120):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = None

    @staticmethod
    def utc(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )

        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    def connect(self):
        if self.session is not None:
            return

        auth = earthaccess.login(strategy="environment")

        if not auth.authenticated:
            raise RuntimeError("Autentikasi NASA Earthdata gagal.")

        self.session = earthaccess.get_requests_https_session()

    @staticmethod
    def grid_indices(west, south, east, north):
        if not (
            -180 <= west < east <= 180
            and -90 <= south < north <= 90
        ):
            raise ValueError("Bounding box IMERG tidak valid.")

        x0 = max(0, math.floor((west + 180) * 10))
        x1 = min(3599, math.ceil((east + 180) * 10) - 1)
        y0 = max(0, math.floor((south + 90) * 10))
        y1 = min(1799, math.ceil((north + 90) * 10) - 1)

        return x0, x1, y0, y1

    @classmethod
    def intervals(cls, start_time, duration):
        start = cls.utc(start_time)
        duration = float(duration)

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Durasi IMERG tidak valid.")

        end = start + timedelta(seconds=duration)

        current = start.replace(
            minute=(start.minute // 30) * 30,
            second=0,
            microsecond=0
        )

        intervals = []

        while current < end:
            intervals.append(current)
            current += timedelta(seconds=cls.INTERVAL)

        return intervals

    @staticmethod
    def granule_time(granule):
        urls = granule.data_links()

        for url in urls:
            match = re.search(
                r"\.(\d{8})-S(\d{6})-E\d{6}",
                url
            )

            if match:
                return datetime.strptime(
                    match.group(1) + match.group(2),
                    "%Y%m%d%H%M%S"
                ).replace(tzinfo=timezone.utc)

        raise ValueError("Waktu granule IMERG tidak ditemukan.")

    @staticmethod
    def opendap_url(granule):
        for link in granule["umm"]["RelatedUrls"]:
            if link.get("Subtype") == "OPENDAP DATA":
                return link["URL"]

        raise ValueError("URL OPeNDAP IMERG tidak ditemukan.")

    def cache_path(self, timestamp, indices):
        key = (
            f"{self.SHORT_NAME}|{self.VERSION}|"
            f"{timestamp.isoformat()}|{indices}"
        )

        digest = hashlib.sha256(key.encode()).hexdigest()

        return self.cache_dir / f"{digest}.nc"

    @staticmethod
    def read_subset(source, indices):
        x0, x1, y0, y1 = indices

        expected_lon = -179.95 + np.arange(x0, x1 + 1) * 0.1
        expected_lat = -89.95 + np.arange(y0, y1 + 1) * 0.1

        dataset = None

        for group in ("Grid", None):
            if hasattr(source, "seek"):
                source.seek(0)

            try:
                dataset = xr.open_dataset(
                    source,
                    engine="h5netcdf",
                    group=group,
                    decode_times=False,
                    mask_and_scale=True
                )
                break
            except (OSError, ValueError, KeyError):
                continue

        if dataset is None:
            raise ValueError("NetCDF IMERG tidak dapat dibaca.")

        with dataset as ds:
            for name in ("precipitation", "lat", "lon", "time"):
                if name not in ds:
                    raise ValueError(
                        f"Variabel IMERG {name} tidak tersedia."
                    )

            rain = ds["precipitation"].transpose(
                "time", "lon", "lat"
            )

            units = rain.attrs.get("units")

            if units != "mm/hr":
                raise ValueError(
                    f"Satuan IMERG tidak sesuai: {units}"
                )

            lat = np.asarray(ds["lat"].values, dtype=float)
            lon = np.asarray(ds["lon"].values, dtype=float)
            values = np.asarray(rain.values, dtype=float)

            expected_shape = (
                1,
                x1 - x0 + 1,
                y1 - y0 + 1
            )

            if values.shape != expected_shape:
                raise ValueError(
                    f"Dimensi IMERG tidak sesuai: {values.shape}"
                )

            if not np.allclose(
                lat, expected_lat, atol=1e-4, rtol=0
            ):
                raise ValueError("Latitude IMERG tidak sesuai.")

            if not np.allclose(
                lon, expected_lon, atol=1e-4, rtol=0
            ):
                raise ValueError("Longitude IMERG tidak sesuai.")

            values[
                ~np.isfinite(values) | (values < 0)
            ] = np.nan

            return {
                "latitude": lat,
                "longitude": lon,
                "precipitation": values[0],
                "units": units
            }

    def download(self, granule, timestamp, indices):
        path = self.cache_path(timestamp, indices)

        if path.exists():
            result = self.read_subset(path, indices)
            print(f"[IMERG CACHE] {timestamp.isoformat()}")
            return result

        self.connect()

        x0, x1, y0, y1 = indices

        constraint = (
            f"/Grid/precipitation[0:1:0]"
            f"[{x0}:1:{x1}][{y0}:1:{y1}];"
            f"/Grid/lon[{x0}:1:{x1}];"
            f"/Grid/lat[{y0}:1:{y1}];"
            "/Grid/time[0:1:0]"
        )

        url = self.opendap_url(granule) + ".dap.nc4"

        with self.session.get(
            url,
            params={"dap4.ce": constraint},
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            content = response.content

        result = self.read_subset(
            BytesIO(content),
            indices
        )

        temporary = path.with_suffix(
            f".{uuid4().hex}.tmp"
        )

        try:
            temporary.write_bytes(content)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

        print(
            f"[IMERG DOWNLOAD] "
            f"{timestamp.isoformat()} | "
            f"{len(content)} bytes",
            flush=True
        )

        return result

    def fetch(self, start_time, duration, bbox):
        indices = self.grid_indices(*bbox)
        intervals = self.intervals(start_time, duration)

        missing = [
            timestamp
            for timestamp in intervals
            if not self.cache_path(timestamp, indices).exists()
        ]

        granules = {}

        if missing:
            self.connect()

            results = earthaccess.search_data(
                short_name=self.SHORT_NAME,
                version=self.VERSION,
                temporal=(
                    missing[0].isoformat(),
                    (
                        missing[-1]
                        + timedelta(seconds=self.INTERVAL)
                    ).isoformat()
                ),
                bounding_box=tuple(bbox)
            )

            for granule in results:
                timestamp = self.granule_time(granule)
                granules[timestamp] = granule

            unavailable = [
                timestamp.isoformat()
                for timestamp in missing
                if timestamp not in granules
            ]

            if unavailable:
                raise ValueError(
                    f"Interval IMERG tidak tersedia: {unavailable}"
                )

        records = []

        for timestamp in intervals:
            record = self.download(
                granules.get(timestamp),
                timestamp,
                indices
            )

            record["start_time"] = timestamp
            records.append(record)

        return records

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
