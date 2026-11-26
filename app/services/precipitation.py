import math
from bisect import bisect_right
from datetime import timedelta

import numpy as np

from app.services.imerg_source import IMERGSource


class IMERGPrecipitationService:
    def __init__(
        self,
        records,
        start_time,
        start_elapsed_seconds,
        duration
    ):
        if not records:
            raise ValueError("Dataset presipitasi IMERG kosong.")

        self.start_time = IMERGSource.utc(start_time)
        self.start_elapsed = float(start_elapsed_seconds)
        self.duration = float(duration)

        records = sorted(
            records,
            key=lambda record: record["start_time"]
        )

        self.latitude = np.asarray(
            records[0]["latitude"], dtype=float
        )

        self.longitude = np.asarray(
            records[0]["longitude"], dtype=float
        )

        self.times = [
            (
                record["start_time"] - self.start_time
            ).total_seconds()
            for record in records
        ]

        self.values = np.stack([
            record["precipitation"]
            for record in records
        ])

        if any(
            b - a != IMERGSource.INTERVAL
            for a, b in zip(self.times, self.times[1:])
        ):
            raise ValueError("Interval waktu IMERG tidak kontinu.")

        if (
            self.times[0] > 0
            or self.times[-1] + IMERGSource.INTERVAL
            < self.duration
        ):
            raise ValueError(
                "Cakupan waktu IMERG tidak mencukupi."
            )

        self.south = round(float(self.latitude[0]) - 0.05, 6)
        self.north = round(float(self.latitude[-1]) + 0.05, 6)
        self.west = round(float(self.longitude[0]) - 0.05, 6)
        self.east = round(float(self.longitude[-1]) + 0.05, 6)

    def rain_at_elapsed(
        self,
        latitude,
        longitude,
        elapsed_seconds
    ):
        lat = float(latitude)
        lon = float(longitude)

        elapsed = (
            float(elapsed_seconds) - self.start_elapsed
        )

        if not all(math.isfinite(v) for v in (lat, lon, elapsed)):
            raise ValueError("Parameter sampling IMERG tidak valid.")

        if elapsed < -1e-7 or elapsed > self.duration + 1e-7:
            raise ValueError("Waktu partikel di luar simulasi.")

        elapsed = min(max(elapsed, 0.0), self.duration)

        if elapsed == self.duration:
            elapsed = math.nextafter(elapsed, -math.inf)

        t = bisect_right(self.times, elapsed) - 1

        if not 0 <= t < len(self.times):
            raise ValueError("Waktu partikel di luar dataset IMERG.")

        if elapsed >= self.times[t] + IMERGSource.INTERVAL:
            raise ValueError("Interval IMERG tidak tersedia.")

        if not (
            self.south <= lat <= self.north
            and self.west <= lon <= self.east
        ):
            raise ValueError(
                f"Partikel di luar domain IMERG: {lat}, {lon}"
            )

        y = min(
            len(self.latitude) - 1,
            math.floor((lat - self.south) * 10)
        )

        x = min(
            len(self.longitude) - 1,
            math.floor((lon - self.west) * 10)
        )

        rain = float(self.values[t, x, y])

        if not math.isfinite(rain) or rain < 0:
            raise ValueError(
                f"Presipitasi IMERG tidak valid: "
                f"lat={lat}, lon={lon}, elapsed={elapsed}"
            )

        return rain

    def diagnostic(self):
        valid = self.values[np.isfinite(self.values)]

        print("\n========== IMERG VALIDATION ==========")
        print("Source          : GPM IMERG Late Run V07")
        print("Start UTC       :", self.start_time.isoformat())
        print(
            "End UTC         :",
            (
                self.start_time
                + timedelta(seconds=self.duration)
            ).isoformat()
        )
        print("Intervals       :", len(self.times))
        print("Resolution      : 0.1 degree")
        print("Time interval   : 1800 seconds")
        print("Units           : mm/hr")
        print("Shape           :", self.values.shape)
        print("Latitude        :", self.south, self.north)
        print("Longitude       :", self.west, self.east)
        print("Missing values  :", int(np.isnan(self.values).sum()))

        if valid.size:
            print("Minimum rain    :", float(valid.min()))
            print("Maximum rain    :", float(valid.max()))
            print("Mean rain       :", float(valid.mean()))

        print("======================================\n", flush=True)
