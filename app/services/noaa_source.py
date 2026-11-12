
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import shutil
import tempfile

import numpy as np
import xarray as xr

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "meteorology"
CACHE_DIR = DATA_DIR / ".noaa_cache"
CACHE_VERSION = "v8"

SOURCE_CONFIG = {
    "GFS": {
        "label": "NOAA GFS 0.25°",
        "filter": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "file_prefix": "gfs",
    },
    "GDAS": {
        "label": "NOAA GDAS 0.25°",
        "filter": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gdas_0p25.pl",
        "file_prefix": "gdas",
    },
}

PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550,
    500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50,
]


def _cycle(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return dt.replace(hour=(dt.hour // 6) * 6)


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "VolcanicAshAnalysis/1.0"})
    try:
        with urlopen(request, timeout=240) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get("Content-Type", "")
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        size = destination.stat().st_size if destination.exists() else 0
        with destination.open("rb") as stream:
            header = stream.read(64)
    except HTTPError as exc:
        body = exc.read(1500).decode("utf-8", "replace")
        raise RuntimeError(
            f"NOAA HTTP {exc.code}\nURL: {url}\nRespons: {body[:1000]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Tidak dapat mengakses NOAA: {exc.reason}\nURL: {url}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Gagal menyimpan respons NOAA: {exc}\nURL: {url}"
        ) from exc

    if status != 200 or size < 10000 or not header.startswith(b"GRIB"):
        sample = header.decode("utf-8", "replace")
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Respons NOAA bukan file GRIB2 yang valid.\n"
            f"URL: {url}\nHTTP: {status}\nContent-Type: {content_type}\n"
            f"Ukuran: {size} byte\nHeader hex: {header.hex()}\n"
            f"Header teks: {sample!r}"
        )


def _query(
    source: str, cycle: datetime, forecast_hour: int,
    lat: float, lon: float, kind: str,
    include_precipitation: bool = True,
) -> str:
    cfg = SOURCE_CONFIG[source]
    prefix = cfg["file_prefix"]
    filename = f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{forecast_hour:03d}"
    directory = f"/{prefix}.{cycle:%Y%m%d}/{cycle:%H}/atmos"

    params = {
        "dir": directory,
        "file": filename,
        "subregion": "",
        "leftlon": f"{max(-180, lon - 15):.2f}",
        "rightlon": f"{min(180, lon + 15):.2f}",
        "toplat": f"{min(90, lat + 12):.2f}",
        "bottomlat": f"{max(-90, lat - 12):.2f}",
    }

    if kind == "upper":
        for level in PRESSURE_LEVELS:
            params[f"lev_{level}_mb"] = "on"
        for variable in ("HGT", "TMP", "UGRD", "VGRD"):
            params[f"var_{variable}"] = "on"
    elif kind == "instant":
        for level in (
            "surface", "2_m_above_ground", "10_m_above_ground",
            "planetary_boundary_layer",
        ):
            params[f"lev_{level}"] = "on"
        for variable in ("TMP", "PRES", "FRICV", "HPBL", "UGRD", "VGRD", "RH"):
            params[f"var_{variable}"] = "on"
    elif kind == "accum":
        params["lev_surface"] = "on"
        params["var_SHTFL"] = "on"
        if include_precipitation:
            params["var_APCP"] = "on"
    else:
        raise ValueError(f"Jenis data tidak dikenal: {kind}")

    return cfg["filter"] + "?" + urlencode(params)


def _open_grib_groups(grib: Path):
    try:
        import cfgrib
    except ImportError as exc:
        raise RuntimeError(
            "GRIB2 membutuhkan cfgrib dan eccodes."
        ) from exc
    try:
        return cfgrib.open_datasets(
            str(grib), backend_kwargs={"indexpath": ""}
        )
    except Exception as exc:
        raise RuntimeError(f"cfgrib gagal membaca {grib.name}: {exc}") from exc


def _merge_groups(grib: Path):
    groups = _open_grib_groups(grib)
    if not groups:
        raise ValueError(f"Tidak ada grup GRIB2 di {grib.name}")
    try:
        datasets = [group.load() for group in groups]
        return xr.merge(datasets, compat="override", join="outer")
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
    return " ".join(str(value).lower() for value in values)


def _is_rh_2m(variable):
    attrs = getattr(variable, "attrs", {})
    short_name = str(attrs.get("GRIB_shortName", "")).lower()
    cf_name = str(attrs.get("GRIB_cfName", "")).lower()
    name = str(attrs.get("GRIB_name", attrs.get("long_name", ""))).lower()
    level_type = str(attrs.get("GRIB_typeOfLevel", "")).lower().replace("_", "")
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
                level_type in {"heightaboveground", "heightabovegroundlayer"}
                and float(level) == 2.0
            )
        except (TypeError, ValueError):
            return False
    return False


def _rename_by_metadata(ds: xr.Dataset, kind: str) -> xr.Dataset:
    aliases = {
        "2t": "t2m", "sp": "sp", "pres": "sp",
        "10u": "u10", "10v": "v10", "gh": "z",
        "hpbl": "blh", "fricv": "zust",
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
            elif "surface pressure" in text and "sp" not in ds.data_vars:
                metadata_rename[name] = "sp"
            elif "friction velocity" in text and "zust" not in ds.data_vars:
                metadata_rename[name] = "zust"
            elif _is_rh_2m(variable) and "rh2m" not in ds.data_vars:
                metadata_rename[name] = "rh2m"

    if metadata_rename:
        ds = ds.rename(metadata_rename)
    return ds


def _ensure_valid_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        return ds
    if "valid_time" in ds.coords and ds["valid_time"].ndim == 0:
        return ds.expand_dims(valid_time=[ds["valid_time"].values])
    if "time" in ds.dims:
        return ds.rename({"time": "valid_time"})
    if "time" in ds.coords and ds["time"].ndim == 0:
        return ds.expand_dims(valid_time=[ds["time"].values])
    raise ValueError("Waktu valid tidak ditemukan di GRIB2.")


def _normalize_precipitation(ds: xr.Dataset) -> xr.Dataset:
    if "tp" not in ds.data_vars:
        template = ds["sshf"]
        ds["tp"] = xr.full_like(template, np.nan, dtype=np.float32)
        ds["tp"].attrs.update({
            "units": "m",
            "long_name": "Total precipitation",
            "availability": "missing",
        })
        return ds

    variable = ds["tp"]
    units = str(variable.attrs.get("units", "")).lower().strip()

    if units in {"kg m**-2", "kg m-2", "kg/m^2", "mm", "millimeter"}:
        attrs = dict(variable.attrs)
        ds["tp"] = variable / 1000.0
        ds["tp"].attrs.update(attrs)
        ds["tp"].attrs["units"] = "m"
    elif units in {"m", "meter", "metre"}:
        ds["tp"].attrs["units"] = "m"
    else:
        raise ValueError(f"Satuan presipitasi tidak dikenal: {units!r}")

    ds["tp"].attrs["availability"] = "available"
    return ds


def _normalize_heat_flux(ds: xr.Dataset) -> xr.Dataset:
    if "sshf" not in ds.data_vars:
        raise ValueError("Sensible heat flux/SHTFL tidak ditemukan.")
    ds["sshf"].attrs.setdefault("units", "J m**-2")
    return ds


def _precipitation_steps(grib: Path):
    import eccodes

    with grib.open("rb") as stream:
        while True:
            gid = eccodes.codes_grib_new_from_file(stream)
            if gid is None:
                break
            try:
                short_name = str(eccodes.codes_get(gid, "shortName")).lower()
                name = str(eccodes.codes_get(gid, "name")).lower()
                step_type = str(eccodes.codes_get(gid, "stepType")).lower()
                is_precip = (
                    short_name in {"tp", "apcp"}
                    or "total precipitation" in name
                )
                if not is_precip or step_type != "accum":
                    continue

                start = float(eccodes.codes_get(gid, "startStep"))
                end = float(eccodes.codes_get(gid, "endStep"))
                step_units = int(eccodes.codes_get(gid, "stepUnits"))
                if step_units != 1:
                    raise ValueError(
                        f"GRIB precipitation stepUnits={step_units}, "
                        "diharapkan satuan jam."
                    )
                if not np.isfinite(start) or not np.isfinite(end):
                    raise ValueError("Interval presipitasi tidak valid.")
                if end <= start:
                    raise ValueError(
                        f"Interval presipitasi tidak valid: {start}-{end}"
                    )
                return start, end
            finally:
                eccodes.codes_release(gid)

    return None


def _convert(grib: Path, dest: Path, kind: str) -> None:
    ds = None
    try:
        if kind == "upper":
            try:
                ds = xr.open_dataset(
                    grib,
                    engine="cfgrib",
                    backend_kwargs={
                        "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
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
                if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
                    ds = ds.rename({"isobaricInhPa": "pressure_level"})

            missing = {"u", "v", "pressure_level"} - (
                set(ds.data_vars) | set(ds.dims) | set(ds.coords)
            )
            if missing:
                raise ValueError(
                    f"Upper-air kehilangan: {sorted(missing)}"
                )

        elif kind == "instant":
            required = {"t2m", "sp", "zust"}
            missing = required - set(ds.data_vars)
            if missing:
                raise ValueError(
                    f"Surface instant kehilangan {sorted(missing)}"
                )
            if "rh2m" in ds.data_vars:
                if not _is_rh_2m(ds["rh2m"]):
                    raise ValueError("Metadata rh2m bukan level 2 meter.")
                ds["rh2m"].attrs["units"] = "%"

        elif kind == "accum":
            ds = _normalize_precipitation(ds)
            ds = _normalize_heat_flux(ds)

        ds = _ensure_valid_time(ds)

        if "latitude" not in ds.variables or "longitude" not in ds.variables:
            raise ValueError("NetCDF tidak memiliki latitude/longitude.")

        if kind == "accum":
            steps = _precipitation_steps(grib)
            if steps is None:
                if ds["tp"].attrs.get("availability") == "available":
                    raise ValueError(
                        "Presipitasi tersedia tetapi interval GRIB "
                        "tidak dapat ditemukan."
                    )
                start, end = np.nan, np.nan
            else:
                start, end = steps

            ds["tp_start_step"] = xr.DataArray(
                [start], dims=("valid_time",),
                attrs={"units": "hours", "long_name": "Precipitation accumulation start"},
            )
            ds["tp_end_step"] = xr.DataArray(
                [end], dims=("valid_time",),
                attrs={"units": "hours", "long_name": "Precipitation accumulation end"},
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(dest)

    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"GRIB2 gagal dikonversi ke NetCDF ({grib.name}): {exc}"
        ) from exc
    finally:
        if ds is not None:
            ds.close()


def _download_convert_cached(
    url: str, dest: Path, kind: str, temp_dir: Path,
) -> None:
    cache_key = hashlib.sha256(
        f"{CACHE_VERSION}|{url}".encode("utf-8")
    ).hexdigest()
    old_key = hashlib.sha256(f"v7|{url}".encode("utf-8")).hexdigest()

    cached_grib = CACHE_DIR / f"{cache_key}.grib2"
    cached_nc = CACHE_DIR / f"{cache_key}.nc"
    old_grib = CACHE_DIR / f"{old_key}.grib2"

    if cached_nc.exists() and cached_nc.stat().st_size > 1000:
        shutil.copy2(cached_nc, dest)
        return

    grib = temp_dir / f"{kind}_{hashlib.md5(url.encode()).hexdigest()[:8]}.grib2"

    valid_cached_grib = False
    for candidate in (cached_grib, old_grib):
        if candidate.exists() and candidate.stat().st_size > 10000:
            with candidate.open("rb") as stream:
                if stream.read(4) == b"GRIB":
                    shutil.copy2(candidate, grib)
                    valid_cached_grib = True
                    break

    if not valid_cached_grib:
        _download(url, grib)

    if not cached_grib.exists():
        shutil.copy2(grib, cached_grib)

    _convert(grib, dest, kind)
    shutil.copy2(dest, cached_nc)


def _combine_forecasts(files, destination, kind):
    datasets = []
    combined = None
    try:
        for path in files:
            with xr.open_dataset(path) as source:
                ds = source.load()

            if "valid_time" not in ds.dims or ds.sizes["valid_time"] != 1:
                raise ValueError(f"Invalid valid_time: {path.name}")

            if "latitude" not in ds.coords or "longitude" not in ds.coords:
                raise ValueError(f"Koordinat tidak tersedia: {path.name}")

            if "step" in ds.coords:
                raw = np.asarray(ds["step"].values).reshape(-1)[0]
                if np.issubdtype(np.asarray(raw).dtype, np.timedelta64):
                    step = float(raw / np.timedelta64(1, "h"))
                else:
                    units = str(ds["step"].attrs.get("units", "")).lower()
                    if units not in {"h", "hr", "hour", "hours"}:
                        raise ValueError(
                            f"Satuan forecast step tidak dikenal: {units!r}"
                        )
                    step = float(raw)
                ds = ds.drop_vars("step")
            else:
                step = float(path.stem.rsplit("_f", 1)[1])

            ds["forecast_step_hours"] = xr.DataArray(
                [step], dims=("valid_time",), attrs={"units": "hours"}
            )

            if "time" in ds.coords:
                reference_time = np.asarray(ds["time"].values).reshape(-1)[0]
                ds.attrs["forecast_reference_time"] = str(reference_time)
                ds = ds.drop_vars("time")

            if datasets:
                first = datasets[0]
                if set(ds.data_vars) != set(first.data_vars):
                    raise ValueError(
                        f"Variabel tidak konsisten: {path.name}\n"
                        f"Hilang: {sorted(set(first.data_vars)-set(ds.data_vars))}\n"
                        f"Tambahan: {sorted(set(ds.data_vars)-set(first.data_vars))}"
                    )
                for coord in ("latitude", "longitude"):
                    if not np.array_equal(ds[coord].values, first[coord].values):
                        raise ValueError(f"Grid {coord} berbeda: {path.name}")
                if kind == "upper":
                    if not np.array_equal(
                        ds["pressure_level"].values,
                        first["pressure_level"].values,
                    ):
                        raise ValueError(f"Pressure level berbeda: {path.name}")

                ref = ds.attrs.get("forecast_reference_time")
                first_ref = first.attrs.get("forecast_reference_time")
                if ref != first_ref:
                    raise ValueError(f"Forecast cycle berbeda: {path.name}")

            datasets.append(ds)

        combined = xr.concat(
            datasets, dim="valid_time",
            join="exact", compat="equals",
            coords="minimal", data_vars="all",
            combine_attrs="override",
        ).sortby("valid_time")

        times = combined["valid_time"].values
        if len(np.unique(times)) != len(times):
            raise ValueError("Duplicate valid_time ditemukan.")
        if len(times) > 1 and not np.all(
            np.diff(times) > np.timedelta64(0, "ns")
        ):
            raise ValueError("Urutan valid_time tidak valid.")

        if kind == "accum":
            combined.attrs["precipitation_rate_status"] = (
                "accumulation_interval_available"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_netcdf(destination)

    finally:
        if combined is not None:
            combined.close()
        for ds in datasets:
            ds.close()


def acquire_noaa_dataset(
    source: str, valid_date: str, forecast_hour: int,
    latitude: float, longitude: float, uploaded_by: int,
    duration_hours: int = 12,
) -> dict:
    source = source.strip().upper()

    if source not in SOURCE_CONFIG:
        raise ValueError("Sumber harus GFS atau GDAS.")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Koordinat tidak valid.")
    if (
        isinstance(forecast_hour, bool)
        or not isinstance(forecast_hour, int)
        or not 0 <= forecast_hour <= 240
        or forecast_hour % 3 != 0
    ):
        raise ValueError("Forecast hour harus 0–240 dengan interval 3 jam.")
    if isinstance(duration_hours, bool) or not isinstance(duration_hours, int):
        raise ValueError("Duration hours harus integer.")
    if not 1 <= duration_hours <= 72:
        raise ValueError("Duration hours harus 1–72 jam.")
    if source == "GDAS" and forecast_hour != 0:
        raise ValueError("Untuk GDAS, gunakan forecast hour 0.")

    try:
        dt = datetime.fromisoformat(valid_date.strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Tanggal harus ISO, contoh 2026-10-27T00:00:00Z."
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    cycle = _cycle(dt.astimezone(timezone.utc))

    if source == "GDAS":
        hours = [0]
    else:
        end_hour = forecast_hour + int(np.ceil(duration_hours / 3)) * 3
        if end_hour > 240:
            raise ValueError("Rentang forecast melewati f240.")
        hours = list(range(forecast_hour, end_hour + 1, 3))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    paths = {}
    created = []

    try:
        with tempfile.TemporaryDirectory(prefix="noaa_") as temp_name:
            temp_dir = Path(temp_name)

            for kind in ("upper", "instant", "accum"):
                forecast_files = []

                for hour in hours:
                    temporary_nc = temp_dir / f"{kind}_f{hour:03d}.nc"

                    if kind == "accum" and source == "GDAS":
                        primary_url = _query(
                            source, cycle, hour, latitude, longitude,
                            kind, include_precipitation=True,
                        )
                        fallback_url = _query(
                            source, cycle, hour, latitude, longitude,
                            kind, include_precipitation=False,
                        )
                        try:
                            _download_convert_cached(
                                primary_url, temporary_nc, kind, temp_dir
                            )
                        except Exception as primary_error:
                            temporary_nc.unlink(missing_ok=True)
                            try:
                                _download_convert_cached(
                                    fallback_url, temporary_nc, kind, temp_dir
                                )
                            except Exception as fallback_error:
                                raise RuntimeError(
                                    f"GDAS accumulated f{hour:03d} gagal.\n"
                                    f"Primary: {primary_error}\n"
                                    f"Fallback: {fallback_error}"
                                ) from fallback_error
                    else:
                        url = _query(
                            source, cycle, hour, latitude, longitude, kind
                        )
                        _download_convert_cached(
                            url, temporary_nc, kind, temp_dir
                        )

                    forecast_files.append(temporary_nc)

                dest = DATA_DIR / f"{source.lower()}_{kind}_{stamp}.nc"
                created.append(dest)
                _combine_forecasts(forecast_files, dest, kind)
                paths[kind] = str(dest.relative_to(BASE_DIR))

    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    return {
        "source": source,
        "cycle": cycle.isoformat(),
        "forecast_hour": forecast_hour,
        "forecast_end_hour": hours[-1],
        "forecast_hours": hours,
        "time_count": len(hours),
        "duration_hours": duration_hours,
        "paths": paths,
        "name": (
            f"{SOURCE_CONFIG[source]['label']} "
            f"{cycle:%Y-%m-%d %H}Z "
            f"f{forecast_hour:03d}-f{hours[-1]:03d}"
        ),
        "uploaded_by": uploaded_by,
    }
