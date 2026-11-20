
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import math
import re
import shutil
import tempfile

import numpy as np
import xarray as xr


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "meteorology"
CACHE_DIR = DATA_DIR / ".noaa_cache"
CACHE_VERSION = "v11-aws-3hour-accum"

AWS_GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

SOURCE_CONFIG = {
    "GFS": {
        "label": "NOAA GFS 0.25° via AWS",
        "file_prefix": "gfs",
    },
    "GDAS": {
        "label": "NOAA GDAS 0.25° via AWS",
        "file_prefix": "gdas",
    },
}

PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 850, 800, 750,
    700, 650, 600, 550, 500, 450, 400, 350,
    300, 250, 200, 150, 100, 70, 50,
]


def _cycle(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    return dt.replace(hour=(dt.hour // 6) * 6)


def _request(url: str, byte_range=None):
    headers = {
        "User-Agent": "WebsiteVolcanic/1.0",
        "Accept-Encoding": "identity",
    }

    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = (
            f"bytes={start}-{end}"
            if end is not None
            else f"bytes={start}-"
        )

    request = Request(url, headers=headers)

    try:
        return urlopen(request, timeout=240)
    except HTTPError as exc:
        body = exc.read(1000).decode("utf-8", "replace")
        raise RuntimeError(
            f"HTTP {exc.code}\nURL: {url}\nRespons: {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Gagal mengakses sumber meteorologi: {exc.reason}\n"
            f"URL: {url}"
        ) from exc


def _validate_grib(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 1000:
        raise ValueError(f"GRIB2 kosong atau terlalu kecil: {path}")

    with path.open("rb") as stream:
        header = stream.read(4)
        stream.seek(-4, 2)
        footer = stream.read(4)

    if header != b"GRIB" or footer != b"7777":
        raise ValueError(f"GRIB2 tidak valid: {path}")


def _aws_gfs_url(
    cycle: datetime,
    forecast_hour: int,
    source: str = "GFS",
) -> str:
    source = source.strip().upper()
    if source not in SOURCE_CONFIG:
        raise ValueError("Sumber harus GFS atau GDAS.")
    if source == "GDAS" and forecast_hour != 0:
        raise ValueError("GDAS hanya menggunakan analisis f000.")
    prefix = SOURCE_CONFIG[source]["file_prefix"]
    return (
        f"{AWS_GFS_BASE}/{prefix}.{cycle:%Y%m%d}/"
        f"{cycle:%H}/atmos/"
        f"{prefix}.t{cycle:%H}z.pgrb2.0p25."
        f"f{forecast_hour:03d}"
    )


def _aws_index(url: str) -> list[dict]:
    with _request(url + ".idx") as response:
        if response.status != 200:
            raise RuntimeError(
                f"Indeks AWS HTTP {response.status}"
            )

        content = response.read().decode("utf-8")

    entries = []

    for line in content.splitlines():
        parts = line.split(":", 4)

        if len(parts) < 5:
            continue

        try:
            offset = int(parts[1])
        except ValueError:
            continue

        entries.append({
            "offset": offset,
            "variable": parts[3].strip().upper(),
            "description": parts[4].strip().upper(),
            "raw": line,
        })

    if not entries:
        raise RuntimeError(
            f"Indeks GRIB2 AWS kosong: {url}.idx"
        )

    entries.sort(key=lambda item: item["offset"])

    for index, entry in enumerate(entries):
        if index + 1 < len(entries):
            next_offset = entries[index + 1]["offset"]

            if next_offset <= entry["offset"]:
                raise ValueError(
                    "Offset indeks AWS tidak valid."
                )

            entry["end"] = next_offset - 1
        else:
            entry["end"] = None

    return entries


def _aws_select(entry: dict, kind: str) -> bool:
    variable = entry["variable"]
    description = entry["description"]
    fields = description.split(":")

    level = fields[0].strip() if fields else ""

    if kind == "upper":
        if variable not in {
            "HGT", "TMP", "UGRD", "VGRD"
        }:
            return False

        match = re.fullmatch(
            r"(\d+)\s+MB",
            level,
            flags=re.IGNORECASE,
        )

        return (
            match is not None
            and int(match.group(1)) in PRESSURE_LEVELS
        )

    if kind == "instant":
        if variable == "TMP":
            return level == "2 M ABOVE GROUND"

        if variable == "PRES":
            return level == "SURFACE"

        if variable == "FRICV":
            return level == "SURFACE"

        if variable == "HPBL":
            return level == "SURFACE"

        if variable in {"UGRD", "VGRD"}:
            return level == "10 M ABOVE GROUND"

        if variable == "RH":
            return level == "2 M ABOVE GROUND"

        return False

    if kind == "accum":
        return (
            variable in {"SHTFL", "APCP"}
            and level == "SURFACE"
        )

    raise ValueError(f"Jenis data tidak dikenal: {kind}")


def _idx_interval(entry: dict):
    match = re.search(r"(?:^|:)(\d+)-(\d+)\s+hour\s+(acc|ave)\s+fcst", entry["description"], re.I)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3).lower()


def _select_accum_messages(entries: list[dict], hour: int) -> list[dict]:
    chosen = []
    for variable, expected in (("APCP", "acc"), ("SHTFL", "ave")):
        candidates = []
        for entry in entries:
            if entry["variable"] != variable:
                continue
            interval = _idx_interval(entry)
            if interval is None or interval[1] != hour or interval[2] != expected:
                continue
            candidates.append((entry, interval))
        if not candidates:
            if variable == "SHTFL":
                raise ValueError(f"SHTFL dengan interval valid tidak tersedia pada f{hour:03d}")
            continue
        if variable == "APCP":
            candidates.sort(key=lambda item: (item[1][0] != 0, item[1][0], item[0]["offset"]))
        else:
            candidates.sort(key=lambda item: (item[1][1] - item[1][0], -item[1][0], item[0]["offset"]))
        chosen.append(candidates[0][0])
    return sorted(chosen, key=lambda entry: entry["offset"])


def _download_aws_grib(
    cycle: datetime,
    forecast_hour: int,
    kind: str,
    destination: Path,
    source: str = "GFS",
    progress_callback=None,
) -> None:
    url = _aws_gfs_url(cycle, forecast_hour, source)
    entries = _aws_index(url)
    selected = [entry for entry in entries if _aws_select(entry, kind)]
    if kind == "accum" and selected:
        selected = _select_accum_messages(selected, forecast_hour)
    if not selected:
        raise RuntimeError(f"Variabel {kind} tidak ditemukan di AWS f{forecast_hour:03d}.")

    selected_names = {entry["variable"] for entry in selected}
    if kind == "upper":
        required = {"HGT", "TMP", "UGRD", "VGRD"}
        for variable in required:
            levels = set()
            for entry in selected:
                if entry["variable"] != variable:
                    continue
                match = re.fullmatch(r"(\d+)\s+MB", entry["description"].split(":")[0].strip())
                if match:
                    levels.add(int(match.group(1)))
            missing_levels = set(PRESSURE_LEVELS) - levels
            if missing_levels:
                raise RuntimeError(f"AWS {variable} kehilangan pressure levels: {sorted(missing_levels)}")
    elif kind == "instant":
        required = {"TMP", "PRES", "FRICV"}
    else:
        required = {"SHTFL"}
    missing = required - selected_names
    if missing:
        raise RuntimeError(f"Variabel AWS tidak lengkap: {sorted(missing)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")

    def fetch_one(item):
        index, entry = item
        start, end = entry["offset"], entry["end"]
        for attempt in range(4):
            try:
                with _request(url, byte_range=(start, end)) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"AWS HTTP Range status {response.status} pada {start}"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(
                        r"bytes\s+(\d+)-(\d+)/(\d+)", content_range
                    )
                    if not match:
                        raise RuntimeError(
                            f"Content-Range AWS tidak valid: {content_range}"
                        )
                    actual_start, actual_end = int(match.group(1)), int(match.group(2))
                    if actual_start != start or (end is not None and actual_end != end):
                        raise RuntimeError(
                            f"HTTP Range offset tidak sesuai: {content_range}"
                        )
                    expected_size = actual_end - actual_start + 1
                    data = response.read()
                if (
                    len(data) != expected_size
                    or not data.startswith(b"GRIB")
                    or not data.endswith(b"7777")
                ):
                    raise RuntimeError(f"Pesan GRIB2 rusak pada offset {start}")
                return index, data
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
                message = str(exc)
                if any(f"HTTP {code}" in message for code in (400, 401, 403, 404, 416)):
                    raise RuntimeError(
                        f"AWS Range {start}-{end} gagal permanen: {exc}"
                    ) from exc
                if attempt == 3:
                    raise RuntimeError(
                        f"AWS Range {start}-{end} gagal setelah 4 percobaan: {exc}"
                    ) from exc
                time.sleep(min(2 ** attempt, 8))

    expected_bytes = (
        sum(item["end"] - item["offset"] + 1 for item in selected)
        if all(item["end"] is not None for item in selected) else 0
    )
    downloaded_bytes = 0
    if progress_callback:
        progress_callback("download", kind, forecast_hour, 0.0, 0, expected_bytes)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            with temporary.open("wb") as output:
                for batch_start in range(0, len(selected), 4):
                    batch = list(enumerate(selected))[batch_start:batch_start + 4]
                    futures = [executor.submit(fetch_one, item) for item in batch]
                    for future in futures:
                        _, data = future.result()
                        output.write(data)
                        downloaded_bytes += len(data)
                        if progress_callback:
                            progress_callback(
                                "download", kind, forecast_hour,
                                (batch_start + futures.index(future) + 1) / len(selected),
                                downloaded_bytes, expected_bytes,
                            )
        _validate_grib(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)



def _subset_gfs(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    if (
        "latitude" not in ds.coords
        or "longitude" not in ds.coords
    ):
        raise ValueError(
            "Koordinat latitude/longitude tidak tersedia."
        )

    lat_min = max(-90.0, latitude - 12.0)
    lat_max = min(90.0, latitude + 12.0)

    lat_values = ds["latitude"].values

    if lat_values[0] > lat_values[-1]:
        ds = ds.sel(
            latitude=slice(lat_max, lat_min)
        )
    else:
        ds = ds.sel(
            latitude=slice(lat_min, lat_max)
        )

    lon_values = ds["longitude"].values
    center_lon = longitude % 360.0

    if float(np.nanmax(lon_values)) <= 180:
        center_lon = longitude

    distance = (
        (ds["longitude"] - center_lon + 180.0) % 360.0
    ) - 180.0

    ds = ds.where(
        np.abs(distance) <= 15.0,
        drop=True,
    )

    if (
        ds.sizes.get("latitude", 0) == 0
        or ds.sizes.get("longitude", 0) == 0
    ):
        raise ValueError(
            "Wilayah GFS tidak mencakup koordinat gunung."
        )

    if float(np.nanmax(ds.longitude.values)) > 180:
        new_lon = (
            (ds.longitude + 180.0) % 360.0
        ) - 180.0

        ds = ds.assign_coords(
            longitude=new_lon
        ).sortby("longitude")

    return ds


def _open_grib_groups(grib: Path):
    try:
        import cfgrib
    except ImportError as exc:
        raise RuntimeError(
            "GRIB2 membutuhkan cfgrib dan eccodes."
        ) from exc

    try:
        return cfgrib.open_datasets(
            str(grib),
            backend_kwargs={"indexpath": ""},
        )
    except Exception as exc:
        raise RuntimeError(
            f"cfgrib gagal membaca {grib.name}: {exc}"
        ) from exc


def _merge_groups(grib: Path):
    groups = _open_grib_groups(grib)

    if not groups:
        raise ValueError(
            f"Tidak ada grup GRIB2 di {grib.name}"
        )

    try:
        datasets = [group.load() for group in groups]

        return xr.merge(
            datasets,
            compat="override",
            join="outer",
        )
    finally:
        for group in groups:
            group.close()


def _metadata_text(variable) -> str:
    values = [
        getattr(variable, "name", ""),
        variable.attrs.get("GRIB_shortName", ""),
        variable.attrs.get("GRIB_name", ""),
        variable.attrs.get("long_name", ""),
        variable.attrs.get("standard_name", ""),
    ]

    return " ".join(
        str(value).lower() for value in values
    )


def _is_rh_2m(variable):
    attrs = getattr(variable, "attrs", {})

    short_name = str(
        attrs.get("GRIB_shortName", "")
    ).lower()

    cf_name = str(
        attrs.get("GRIB_cfName", "")
    ).lower()

    name = str(
        attrs.get(
            "GRIB_name",
            attrs.get("long_name", ""),
        )
    ).lower()

    level_type = str(
        attrs.get("GRIB_typeOfLevel", "")
    ).lower().replace("_", "")

    level = attrs.get("GRIB_level")

    is_rh = (
        short_name in {"2r", "r2", "r"}
        or cf_name == "relative_humidity"
        or "relative humidity" in name
    )

    if not is_rh:
        return False

    if short_name in {"2r", "r2"}:
        return True

    if level is not None:
        try:
            return (
                level_type in {
                    "heightaboveground",
                    "heightabovegroundlayer",
                }
                and float(level) == 2.0
            )
        except (TypeError, ValueError):
            return False

    return False


def _rename_by_metadata(
    ds: xr.Dataset,
    kind: str,
) -> xr.Dataset:
    aliases = {
        "2t": "t2m",
        "sp": "sp",
        "pres": "sp",
        "10u": "u10",
        "10v": "v10",
        "gh": "z",
        "hpbl": "blh",
        "fricv": "zust",
    }

    rename = {}

    for name in ds.data_vars:
        target = aliases.get(name.lower())

        if target and target not in ds.data_vars:
            rename[name] = target

    if rename:
        ds = ds.rename(rename)

    metadata_rename = {}

    for name in ds.data_vars:
        variable = ds[name]
        text = _metadata_text(variable)
        lower = name.lower()

        if kind == "accum":
            if (
                lower in {"apcp", "tp"}
                or "total precipitation" in text
                or "precipitation amount" in text
            ) and "tp" not in ds.data_vars:
                metadata_rename[name] = "tp"

            elif (
                lower in {"shtfl", "sshf"}
                or "sensible heat flux" in text
            ) and "sshf" not in ds.data_vars:
                metadata_rename[name] = "sshf"

        elif kind == "instant":
            if (
                lower == "2t"
                or "2 metre temperature" in text
                or "2 meter temperature" in text
            ) and "t2m" not in ds.data_vars:
                metadata_rename[name] = "t2m"

            elif (
                "surface pressure" in text
                and "sp" not in ds.data_vars
            ):
                metadata_rename[name] = "sp"

            elif (
                "friction velocity" in text
                and "zust" not in ds.data_vars
            ):
                metadata_rename[name] = "zust"

            elif (
                _is_rh_2m(variable)
                and "rh2m" not in ds.data_vars
            ):
                metadata_rename[name] = "rh2m"

    if metadata_rename:
        ds = ds.rename(metadata_rename)

    return ds


def _ensure_valid_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        return ds

    if (
        "valid_time" in ds.coords
        and ds["valid_time"].ndim == 0
    ):
        return ds.expand_dims(
            valid_time=[ds["valid_time"].values]
        )

    if "time" in ds.dims:
        return ds.rename({"time": "valid_time"})

    if (
        "time" in ds.coords
        and ds["time"].ndim == 0
    ):
        return ds.expand_dims(
            valid_time=[ds["time"].values]
        )

    raise ValueError(
        "Waktu valid tidak ditemukan di GRIB2."
    )


def _normalize_precipitation(
    ds: xr.Dataset,
) -> xr.Dataset:
    if "tp" not in ds.data_vars:
        template = ds["sshf"]

        ds["tp"] = xr.full_like(
            template,
            np.nan,
            dtype=np.float32,
        )

        ds["tp"].attrs.update({
            "units": "m",
            "long_name": "Total precipitation",
            "availability": "missing",
        })

        return ds

    variable = ds["tp"]
    units = str(
        variable.attrs.get("units", "")
    ).lower().strip()

    if units in {
        "kg m**-2",
        "kg m-2",
        "kg/m^2",
        "mm",
        "millimeter",
        "millimetre",
    }:
        attrs = dict(variable.attrs)

        ds["tp"] = variable / 1000.0
        ds["tp"].attrs.update(attrs)
        ds["tp"].attrs["units"] = "m"

    elif units in {"m", "meter", "metre"}:
        ds["tp"].attrs["units"] = "m"

    else:
        raise ValueError(
            f"Satuan presipitasi tidak dikenal: {units!r}"
        )

    ds["tp"].attrs["availability"] = "available"

    return ds


def _normalize_heat_flux(
    ds: xr.Dataset,
) -> xr.Dataset:
    if "sshf" not in ds.data_vars:
        raise ValueError(
            "Sensible heat flux/SHTFL tidak ditemukan."
        )

    units = str(
        ds["sshf"].attrs.get("units", "")
    ).strip()

    if not units:
        raise ValueError(
            "Satuan sensible heat flux tidak tersedia."
        )

    ds["sshf"].attrs["source_units"] = units

    return ds


def _accumulation_steps(grib: Path) -> dict:
    import eccodes
    result = {}
    with grib.open("rb") as stream:
        while True:
            gid = eccodes.codes_grib_new_from_file(stream)
            if gid is None:
                break
            try:
                short = str(eccodes.codes_get(gid, "shortName")).lower()
                name = str(eccodes.codes_get(gid, "name")).lower()
                kind = str(eccodes.codes_get(gid, "stepType")).lower()
                variable = None
                if short in {"tp", "apcp"} or "total precipitation" in name:
                    variable = "tp"
                    expected = "accum"
                elif short in {"shtfl", "sshf"} or "sensible heat flux" in name:
                    variable = "sshf"
                    expected = "avg"
                if variable is None:
                    continue
                if kind not in ({"accum"} if variable == "tp" else {"avg", "average"}):
                    raise ValueError(f"Jenis interval {variable} tidak sesuai: {kind}")
                try:
                    start = float(eccodes.codes_get(gid, "startStep"))
                    end = float(eccodes.codes_get(gid, "endStep"))
                except (TypeError, ValueError):
                    step_range = str(eccodes.codes_get(gid, "stepRange"))
                    match = re.fullmatch(r"(\d+)-(\d+)", step_range)
                    if not match:
                        raise ValueError(f"Interval {variable} tidak terbaca: {step_range!r}")
                    start, end = map(float, match.groups())
                step_units = int(eccodes.codes_get(gid, "stepUnits"))
                if step_units != 1 or not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end):
                    raise ValueError(f"Interval {variable} tidak valid: {start}-{end}, units={step_units}")
                if variable in result:
                    raise ValueError(f"Duplikasi pesan {variable} pada GRIB terpilih")
                result[variable] = (start, end)
            finally:
                eccodes.codes_release(gid)
    if "sshf" not in result:
        raise ValueError("Metadata interval SHTFL tidak ditemukan")
    return result


def _convert(
    grib: Path,
    dest: Path,
    kind: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> None:
    ds = None

    try:
        if kind == "upper":
            try:
                ds = xr.open_dataset(
                    grib,
                    engine="cfgrib",
                    backend_kwargs={
                        "filter_by_keys": {
                            "typeOfLevel": "isobaricInhPa"
                        },
                        "indexpath": "",
                    },
                ).load()
            except Exception:
                ds = _merge_groups(grib)

        else:
            ds = _merge_groups(grib)

        ds = _rename_by_metadata(ds, kind)

        if kind == "upper":
            if "pressure_level" not in ds.dims:
                if (
                    "isobaricInhPa" in ds.dims
                    or "isobaricInhPa" in ds.coords
                ):
                    ds = ds.rename({
                        "isobaricInhPa": "pressure_level"
                    })

            required = {
                "u", "v", "t", "z", "pressure_level"
            }

            available = (
                set(ds.data_vars)
                | set(ds.dims)
                | set(ds.coords)
            )

            missing = required - available

            if missing:
                raise ValueError(
                    f"Upper-air kehilangan: {sorted(missing)}"
                )

        elif kind == "instant":
            required = {"t2m", "sp", "zust"}
            missing = required - set(ds.data_vars)

            if missing:
                raise ValueError(
                    f"Surface instant kehilangan: "
                    f"{sorted(missing)}"
                )

            if "rh2m" in ds.data_vars:
                if not _is_rh_2m(ds["rh2m"]):
                    raise ValueError(
                        "Metadata rh2m bukan level 2 meter."
                    )

                ds["rh2m"].attrs["units"] = "%"

        elif kind == "accum":
            ds = _normalize_precipitation(ds)
            ds = _normalize_heat_flux(ds)

        else:
            raise ValueError(
                f"Jenis konversi tidak dikenal: {kind}"
            )

        ds = _ensure_valid_time(ds)

        if (
            latitude is not None
            and longitude is not None
        ):
            ds = _subset_gfs(
                ds, latitude, longitude
            )

        if (
            "latitude" not in ds.variables
            or "longitude" not in ds.variables
        ):
            raise ValueError(
                "NetCDF tidak memiliki latitude/longitude."
            )

        if kind == "accum":
            steps = _accumulation_steps(grib)
            if ds["tp"].attrs.get("availability") == "available" and "tp" not in steps:
                raise ValueError("APCP tersedia tetapi interval akumulasi tidak ditemukan")
            for variable in ("tp", "sshf"):
                start_step, end_step = steps.get(variable, (np.nan, np.nan))
                for suffix, value in (("start_step", start_step), ("end_step", end_step)):
                    ds[f"{variable}_{suffix}"] = xr.DataArray(
                        [value], dims=("valid_time",), attrs={"units": "hours"}
                    )

        dest.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ds.to_netcdf(dest)

    except Exception as exc:
        dest.unlink(missing_ok=True)

        raise RuntimeError(
            f"GRIB2 gagal dikonversi ke NetCDF "
            f"({grib.name}): {exc}"
        ) from exc

    finally:
        if ds is not None:
            ds.close()


def _cache_key(
    source: str,
    url: str,
    kind: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> str:
    identity = (
        f"{CACHE_VERSION}|{source}|{url}|{kind}|"
        f"{latitude}|{longitude}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def _download_convert_aws_cached(
    cycle: datetime,
    forecast_hour: int,
    kind: str,
    latitude: float,
    longitude: float,
    destination: Path,
    temp_dir: Path,
    source: str = "GFS",
    progress_callback=None,
) -> None:
    source = source.strip().upper()
    url = _aws_gfs_url(cycle, forecast_hour, source)
    cache_key = _cache_key(
        f"AWS-{source}", url, kind, latitude, longitude
    )
    cached_nc = CACHE_DIR / f"{cache_key}.nc"
    grib_key = _cache_key(f"AWS-{source}-GRIB", url, kind)
    cached_grib = CACHE_DIR / f"{grib_key}.grib2"

    if cached_nc.exists() and cached_nc.stat().st_size > 1000:
        shutil.copy2(cached_nc, destination)
        if progress_callback:
            progress_callback("cache", kind, forecast_hour, 1.0, 0, 0)
        return

    if cached_grib.exists():
        try:
            _validate_grib(cached_grib)
        except ValueError:
            cached_grib.unlink(missing_ok=True)

    if not cached_grib.exists():
        _download_aws_grib(
            cycle, forecast_hour, kind, cached_grib, source=source,
            progress_callback=progress_callback,
        )

    temporary_nc = temp_dir / (
        f"aws_{source.lower()}_{kind}_f{forecast_hour:03d}.nc"
    )
    if progress_callback:
        progress_callback("convert", kind, forecast_hour, 0.0, 0, 0)
    _convert(
        cached_grib,
        temporary_nc,
        kind,
        latitude=latitude,
        longitude=longitude,
    )
    shutil.copy2(temporary_nc, destination)
    shutil.copy2(temporary_nc, cached_nc)
    if progress_callback:
        progress_callback("convert", kind, forecast_hour, 1.0, 0, 0)


def _normalize_three_hour_accum(files: list[Path]) -> None:
    previous = {"tp": {}, "sshf": {}}
    for path in files:
        with xr.open_dataset(path) as opened:
            ds = opened.load()
        try:
            step = float(np.asarray(ds["step"].values).reshape(-1)[0] / np.timedelta64(1, "h")) if np.issubdtype(ds["step"].dtype, np.timedelta64) else float(np.asarray(ds["step"].values).reshape(-1)[0])
            if step <= 0 or step % 3 != 0:
                raise ValueError(f"Forecast accum harus kelipatan 3 jam: {path.name}")
            left, right = int(step - 3), int(step)
            for variable in ("tp", "sshf"):
                if variable == "tp" and ds["tp"].attrs.get("availability") == "missing":
                    continue
                start = float(ds[f"{variable}_start_step"].values[0])
                end = float(ds[f"{variable}_end_step"].values[0])
                if not (np.isfinite(start) and np.isfinite(end) and start < end and abs(end - right) < 1e-6):
                    raise ValueError(f"Interval {variable} tidak cocok f{right:03d}: {start}-{end}")
                if abs(start / 3 - round(start / 3)) > 1e-6:
                    raise ValueError(f"Interval {variable} tidak sejajar 3 jam: {start}-{end}")
                attrs = dict(ds[variable].attrs)
                source = np.asarray(ds[variable].values, dtype=np.float64)
                interval = end - start
                amount = source.copy() if variable == "tp" else source * interval
                for hour in range(int(start), left, 3):
                    if hour not in previous[variable]:
                        raise ValueError(f"Tidak bisa membentuk interval {left}-{right} {variable}; data {hour}-{hour+3} tidak ada")
                    amount = amount - previous[variable][hour]
                if variable == "tp":
                    if np.any(amount < -1e-7):
                        raise ValueError(f"Akumulasi APCP negatif pada {left}-{right}; periksa produk NOAA")
                    amount = np.maximum(amount, 0)
                    normalized = amount
                    previous[variable][left] = amount.copy()
                else:
                    normalized = amount / 3.0
                    previous[variable][left] = amount.copy()
                ds[variable].values = normalized.astype(ds[variable].dtype, copy=False)
                ds[variable].attrs = attrs
                ds[variable].attrs["normalized_interval_hours"] = 3
                ds[variable].attrs["source_interval_start_hour"] = start
                ds[variable].attrs["source_interval_end_hour"] = end
                ds[f"{variable}_start_step"] = xr.DataArray([float(left)], dims=("valid_time",), attrs={"units": "hours"})
                ds[f"{variable}_end_step"] = xr.DataArray([float(right)], dims=("valid_time",), attrs={"units": "hours"})
            temporary = path.with_suffix(".normalized.nc")
            ds.to_netcdf(temporary)
            temporary.replace(path)
        finally:
            ds.close()


def _combine_forecasts(
    files,
    destination,
    kind,
):
    datasets = []
    combined = None

    try:
        for path in files:
            with xr.open_dataset(path) as source:
                ds = source.load()

            if (
                "valid_time" not in ds.dims
                or ds.sizes["valid_time"] != 1
            ):
                raise ValueError(
                    f"Invalid valid_time: {path.name}"
                )

            if (
                "latitude" not in ds.coords
                or "longitude" not in ds.coords
            ):
                raise ValueError(
                    f"Koordinat tidak tersedia: {path.name}"
                )

            if "step" in ds.coords:
                raw = np.asarray(
                    ds["step"].values
                ).reshape(-1)[0]

                if np.issubdtype(
                    np.asarray(raw).dtype,
                    np.timedelta64,
                ):
                    step = float(
                        raw / np.timedelta64(1, "h")
                    )
                else:
                    units = str(
                        ds["step"].attrs.get("units", "")
                    ).lower()

                    if units not in {
                        "h", "hr", "hour", "hours"
                    }:
                        raise ValueError(
                            "Satuan forecast step "
                            f"tidak dikenal: {units!r}"
                        )

                    step = float(raw)

                ds = ds.drop_vars("step")

            else:
                step = float(
                    path.stem.rsplit("_f", 1)[1]
                )

            ds["forecast_step_hours"] = xr.DataArray(
                [step],
                dims=("valid_time",),
                attrs={"units": "hours"},
            )

            if "time" in ds.coords:
                reference_time = np.asarray(
                    ds["time"].values
                ).reshape(-1)[0]

                ds.attrs["forecast_reference_time"] = (
                    str(reference_time)
                )

                ds = ds.drop_vars("time")

            if datasets:
                first = datasets[0]

                if (
                    set(ds.data_vars)
                    != set(first.data_vars)
                ):
                    raise ValueError(
                        f"Variabel tidak konsisten: "
                        f"{path.name}\n"
                        f"Hilang: "
                        f"{sorted(set(first.data_vars) - set(ds.data_vars))}\n"
                        f"Tambahan: "
                        f"{sorted(set(ds.data_vars) - set(first.data_vars))}"
                    )

                for coord in (
                    "latitude",
                    "longitude",
                ):
                    if not np.array_equal(
                        ds[coord].values,
                        first[coord].values,
                    ):
                        raise ValueError(
                            f"Grid {coord} berbeda: {path.name}"
                        )

                if kind == "upper":
                    if not np.array_equal(
                        ds["pressure_level"].values,
                        first["pressure_level"].values,
                    ):
                        raise ValueError(
                            f"Pressure level berbeda: {path.name}"
                        )

                ref = ds.attrs.get(
                    "forecast_reference_time"
                )

                first_ref = first.attrs.get(
                    "forecast_reference_time"
                )

                if ref != first_ref:
                    raise ValueError(
                        f"Forecast cycle berbeda: {path.name}"
                    )

            datasets.append(ds)

        combined = xr.concat(
            datasets,
            dim="valid_time",
            join="exact",
            compat="equals",
            coords="minimal",
            data_vars="all",
            combine_attrs="override",
        ).sortby("valid_time")

        times = combined["valid_time"].values

        if len(np.unique(times)) != len(times):
            raise ValueError(
                "Duplicate valid_time ditemukan."
            )

        if (
            len(times) > 1
            and not np.all(
                np.diff(times)
                > np.timedelta64(0, "ns")
            )
        ):
            raise ValueError(
                "Urutan valid_time tidak valid."
            )

        if kind == "accum":
            combined.attrs[
                "precipitation_rate_status"
            ] = "accumulation_interval_available"

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined.to_netcdf(destination)

    finally:
        if combined is not None:
            combined.close()

        for ds in datasets:
            ds.close()


def _download_gfs(
    cycle: datetime,
    forecast_hour: int,
    latitude: float,
    longitude: float,
    kind: str,
    destination: Path,
    temp_dir: Path,
    progress_callback=None,
) -> str:
    _download_convert_aws_cached(
        cycle=cycle,
        forecast_hour=forecast_hour,
        kind=kind,
        latitude=latitude,
        longitude=longitude,
        destination=destination,
        temp_dir=temp_dir,
        source="GFS",
        progress_callback=progress_callback,
    )
    return "AWS"


def _download_gdas(
    cycle: datetime,
    forecast_hour: int,
    latitude: float,
    longitude: float,
    kind: str,
    destination: Path,
    temp_dir: Path,
    progress_callback=None,
) -> str:
    _download_convert_aws_cached(
        cycle=cycle,
        forecast_hour=forecast_hour,
        kind=kind,
        latitude=latitude,
        longitude=longitude,
        destination=destination,
        temp_dir=temp_dir,
        source="GDAS",
        progress_callback=progress_callback,
    )
    return "AWS"


def acquire_noaa_dataset(
    source: str,
    valid_date: str,
    forecast_hour: int,
    latitude: float,
    longitude: float,
    uploaded_by: int,
    duration_hours: int = 12,
    progress_callback=None,
) -> dict:
    source = source.strip().upper()

    if source not in SOURCE_CONFIG:
        raise ValueError(
            "Sumber harus GFS atau GDAS."
        )

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError(
            "Koordinat tidak valid."
        )

    if (
        isinstance(forecast_hour, bool)
        or not isinstance(forecast_hour, int)
        or not 0 <= forecast_hour <= 240
        or forecast_hour % 3 != 0
    ):
        raise ValueError(
            "Forecast hour harus 0–240 "
            "dengan interval 3 jam."
        )

    if (
        isinstance(duration_hours, bool)
        or not isinstance(duration_hours, int)
    ):
        raise ValueError(
            "Duration hours harus integer."
        )

    if not 1 <= duration_hours <= 72:
        raise ValueError(
            "Duration hours harus 1–72 jam."
        )

    if source == "GDAS" and forecast_hour != 0:
        raise ValueError(
            "Untuk GDAS, gunakan forecast hour 0."
        )

    try:
        dt = datetime.fromisoformat(
            valid_date.strip().replace(
                "Z", "+00:00"
            )
        )
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Tanggal harus ISO, contoh "
            "2026-11-02T06:00:00Z."
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    cycle = _cycle(
        dt.astimezone(timezone.utc)
    )

    if source == "GDAS":
        hours = [0]
    else:
        end_hour = (
            forecast_hour
            + int(np.ceil(duration_hours / 3)) * 3
        )

        if end_hour > 240:
            raise ValueError(
                "Rentang forecast melewati f240."
            )

        hours = list(
            range(
                forecast_hour,
                end_hour + 1,
                3,
            )
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d%H%M%S%f")

    paths = {}
    created = []
    def emit(stage, kind=None, hour=None, fraction=0.0, received=0, total=0):
        if progress_callback:
            progress_callback({
                "stage": stage, "kind": kind, "forecast_hour": hour,
                "fraction": max(0.0, min(1.0, float(fraction))),
                "bytes_received": int(received), "bytes_total": int(total),
            })
    planned = {
        kind: (list(range(3, hours[-1] + 1, 3)) if kind == "accum" and source == "GFS" else hours)
        for kind in ("upper", "instant", "accum")
    }
    total_units = sum(2 * len(items) + 1 for items in planned.values()) + 1
    finished_units = 0
    def item_progress(stage, kind, hour, fraction, received, total):
        base = finished_units
        if stage == "cache":
            value = 2.0
        elif stage == "convert":
            value = 1.0 + fraction
        else:
            value = fraction
        emit(stage, kind, hour, (base + value) / total_units, received, total)
    emit("starting", fraction=0.0)
    download_sources = {}

    try:
        with tempfile.TemporaryDirectory(
            prefix="gfs_"
        ) as temp_name:
            temp_dir = Path(temp_name)

            for kind in (
                "upper",
                "instant",
                "accum",
            ):
                forecast_files = []
                download_sources[kind] = {}

                kind_hours = (list(range(3, hours[-1] + 1, 3)) if kind == "accum" and source == "GFS" else hours)
                if kind == "accum" and not kind_hours:
                    raise ValueError("Surface Accum memerlukan minimal f003; rentang waktu tidak cukup")
                for hour in kind_hours:
                    temporary_nc = (
                        temp_dir
                        / f"{kind}_f{hour:03d}.nc"
                    )

                    emit("preparing", kind, hour, finished_units / total_units)
                    print(
                        f"[{source}] "
                        f"{cycle:%Y-%m-%d %H}Z "
                        f"f{hour:03d} "
                        f"kind={kind}"
                    )

                    if source == "GFS":
                        provider = _download_gfs(
                            cycle=cycle,
                            forecast_hour=hour,
                            latitude=latitude,
                            longitude=longitude,
                            kind=kind,
                            destination=temporary_nc,
                            temp_dir=temp_dir,
                            progress_callback=item_progress,
                        )
                    else:
                        provider = _download_gdas(
                            cycle=cycle,
                            forecast_hour=hour,
                            latitude=latitude,
                            longitude=longitude,
                            kind=kind,
                            destination=temporary_nc,
                            temp_dir=temp_dir,
                            progress_callback=item_progress,
                        )

                    finished_units += 2
                    emit("forecast_completed", kind, hour, finished_units / total_units)
                    download_sources[kind][
                        f"f{hour:03d}"
                    ] = provider

                    forecast_files.append(
                        temporary_nc
                    )

                    print(
                        f"[{source}] "
                        f"f{hour:03d} "
                        f"{kind} selesai "
                        f"via {provider}"
                    )

                dest = DATA_DIR / (
                    f"{source.lower()}_"
                    f"{kind}_{stamp}.nc"
                )

                created.append(dest)

                if kind == "accum" and source == "GFS":
                    _normalize_three_hour_accum(forecast_files)
                    forecast_files = [
                        path for hour, path in zip(kind_hours, forecast_files)
                        if hour in hours
                    ]
                emit("combining", kind, None, finished_units / total_units)
                _combine_forecasts(
                    forecast_files,
                    dest,
                    kind,
                )

                finished_units += 1
                emit("combined", kind, None, finished_units / total_units)
                paths[kind] = str(
                    dest.relative_to(BASE_DIR)
                )

    except Exception:
        for path in created:
            path.unlink(
                missing_ok=True
            )

        raise

    emit("download_completed", fraction=finished_units / total_units)
    return {
        "source": source,
        "cycle": cycle.isoformat(),
        "forecast_hour": forecast_hour,
        "forecast_end_hour": hours[-1],
        "forecast_hours": hours,
        "time_count": len(hours),
        "duration_hours": duration_hours,
        "paths": paths,
        "download_sources": download_sources,
        "name": (
            f"{SOURCE_CONFIG[source]['label']} "
            f"{cycle:%Y-%m-%d %H}Z "
            f"f{forecast_hour:03d}-"
            f"f{hours[-1]:03d}"
        ),
        "uploaded_by": uploaded_by,
    }
