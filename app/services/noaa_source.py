
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
CACHE_VERSION = "v4"

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
    dt = dt.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    return dt.replace(hour=(dt.hour // 6) * 6)


def _download(url: str, destination: Path) -> None:
    request = Request(
        url,
        headers={"User-Agent": "VolcanicAshAnalysis/1.0"},
    )

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
            f"NOAA HTTP {exc.code}\nURL: {url}\n"
            f"Respons: {body[:1000]}"
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
            f"URL: {url}\n"
            f"HTTP: {status}\n"
            f"Content-Type: {content_type}\n"
            f"Ukuran: {size} byte\n"
            f"Header hex: {header.hex()}\n"
            f"Header teks: {sample!r}"
        )


def _query(
    source: str,
    cycle: datetime,
    forecast_hour: int,
    lat: float,
    lon: float,
    kind: str,
) -> str:
    cfg = SOURCE_CONFIG[source]
    prefix = cfg["file_prefix"]
    filename = f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{forecast_hour:03d}"
    directory = f"/{prefix}.{cycle:%Y%m%d}/{cycle:%H}/atmos"

    west = max(-180, lon - 15)
    east = min(180, lon + 15)
    south = max(-90, lat - 12)
    north = min(90, lat + 12)

    params = {
        "dir": directory,
        "file": filename,
        "subregion": "",
        "leftlon": f"{west:.2f}",
        "rightlon": f"{east:.2f}",
        "toplat": f"{north:.2f}",
        "bottomlat": f"{south:.2f}",
    }

    if kind == "upper":
        for level in PRESSURE_LEVELS:
            params[f"lev_{level}_mb"] = "on"
        for variable in ("HGT", "TMP", "UGRD", "VGRD"):
            params[f"var_{variable}"] = "on"

    elif kind == "instant":
        for level in (
            "surface",
            "2_m_above_ground",
            "10_m_above_ground",
            "planetary_boundary_layer",
        ):
            params[f"lev_{level}"] = "on"
        for variable in ("TMP", "PRES", "FRICV", "HPBL", "UGRD", "VGRD"):
            params[f"var_{variable}"] = "on"

    elif kind == "accum":
        params["lev_surface"] = "on"
        params["var_SHTFL"] = "on"
        if forecast_hour > 0:
            params["var_APCP"] = "on"

    else:
        raise ValueError(f"Jenis data tidak dikenal: {kind}")

    return cfg["filter"] + "?" + urlencode(params)


def _open_grib_groups(grib: Path):
    try:
        import cfgrib
    except ImportError as exc:
        raise RuntimeError(
            "GRIB2 membutuhkan cfgrib dan eccodes. "
            "Instal keduanya pada virtual environment aplikasi."
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
        raise ValueError(f"Tidak ada grup GRIB2 di {grib.name}")

    try:
        datasets = [group.load() for group in groups]
        return xr.merge(datasets, compat="override", join="outer")
    finally:
        for group in groups:
            try:
                group.close()
            except Exception:
                pass


def _metadata_text(variable) -> str:
    values = [
        getattr(variable, "name", ""),
        variable.attrs.get("GRIB_shortName", ""),
        variable.attrs.get("GRIB_name", ""),
        variable.attrs.get("long_name", ""),
        variable.attrs.get("standard_name", ""),
    ]
    return " ".join(str(value).lower() for value in values)


def _rename_by_metadata(ds: xr.Dataset, kind: str) -> xr.Dataset:
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
        lower = name.lower()
        target = aliases.get(lower)
        if target and target not in ds.data_vars:
            rename[name] = target

    if rename:
        ds = ds.rename(rename)

    metadata_rename = {}
    for name in ds.data_vars:
        text = _metadata_text(ds[name])
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
        available = {
            name: _metadata_text(ds[name]) for name in ds.data_vars
        }
        raise ValueError(
            "Variabel total precipitation/APCP tidak ditemukan. "
            f"Variabel tersedia: {available}"
        )

    variable = ds["tp"]
    units = str(variable.attrs.get("units", "")).lower().strip()

    if units in {"kg m**-2", "kg m-2", "kg/m^2", "mm", "millimeter"}:
        ds["tp"] = variable / 1000.0
        ds["tp"].attrs.update({
            "units": "m",
            "long_name": "Total precipitation",
        })
    elif units in {"m", "meter", "metre"}:
        ds["tp"].attrs["units"] = "m"
    else:
        raise ValueError(
            f"Satuan presipitasi tidak dikenal: {units!r}. "
            "Tidak dilakukan konversi otomatis."
        )

    return ds


def _normalize_heat_flux(ds: xr.Dataset) -> xr.Dataset:
    if "sshf" not in ds.data_vars:
        available = {
            name: _metadata_text(ds[name]) for name in ds.data_vars
        }
        raise ValueError(
            "Sensible heat flux/SHTFL tidak ditemukan. "
            f"Variabel tersedia: {available}"
        )

    ds["sshf"].attrs.setdefault("units", "J m**-2")
    return ds


def _create_zero_accum(dest: Path, instant_path: Path) -> None:
    """Buat nilai awal nol pada grid yang sama dengan data instant."""
    with xr.open_dataset(instant_path) as instant:
        template = instant["t2m"].load()
        ds = xr.Dataset({
            "tp": xr.zeros_like(template, dtype=np.float32),
            "sshf": xr.zeros_like(template, dtype=np.float32),
        })
        ds["tp"].attrs.update({
            "units": "m",
            "long_name": "Total precipitation at forecast hour 0",
        })
        ds["sshf"].attrs.update({
            "units": "J m**-2",
            "long_name": "Sensible heat flux accumulation at forecast hour 0",
        })
        ds = _ensure_valid_time(ds)
        ds.to_netcdf(dest)


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
                if (
                    "isobaricInhPa" in ds.dims
                    or "isobaricInhPa" in ds.coords
                ):
                    ds = ds.rename({"isobaricInhPa": "pressure_level"})

            missing = {"u", "v", "pressure_level"} - (
                set(ds.data_vars) | set(ds.dims) | set(ds.coords)
            )
            if missing:
                raise ValueError(
                    f"Upper-air tidak memiliki variabel/dimensi: {sorted(missing)}"
                )

        elif kind == "instant":
            required = {"t2m", "sp", "zust"}
            missing = required - set(ds.data_vars)
            if missing:
                available = {
                    name: _metadata_text(ds[name]) for name in ds.data_vars
                }
                raise ValueError(
                    f"Surface instant kehilangan {sorted(missing)}. "
                    f"Variabel tersedia: {available}"
                )

        elif kind == "accum":
            ds = _normalize_precipitation(ds)
            ds = _normalize_heat_flux(ds)

        ds = _ensure_valid_time(ds)
        if "latitude" not in ds.variables or "longitude" not in ds.variables:
            raise ValueError(
                "NetCDF hasil konversi tidak memiliki latitude/longitude."
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
            try:
                ds.close()
            except Exception:
                pass


def acquire_noaa_dataset(
    source: str,
    valid_date: str,
    forecast_hour: int,
    latitude: float,
    longitude: float,
    uploaded_by: int,
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

    try:
        dt = datetime.fromisoformat(
            valid_date.strip().replace("Z", "+00:00")
        )
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Tanggal harus ISO, contoh 2026-10-27T00:00:00Z."
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    if source == "GDAS" and forecast_hour != 0:
        raise ValueError("Untuk GDAS, gunakan forecast hour 0.")

    cycle = _cycle(dt)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    paths = {}
    instant_path = None

    try:
        with tempfile.TemporaryDirectory(prefix="noaa_") as temp_name:
            temp_dir = Path(temp_name)

            for kind in ("upper", "instant", "accum"):
                dest = DATA_DIR / f"{source.lower()}_{kind}_{stamp}.nc"

                if kind == "accum" and forecast_hour == 0:
                    if instant_path is None:
                        raise RuntimeError(
                            "Data instant harus tersedia sebelum membuat "
                            "data akumulasi awal."
                        )
                    cache_key = hashlib.sha256(
                        f"{CACHE_VERSION}|{source}|{cycle.isoformat()}|"
                        f"{forecast_hour}|{latitude:.4f}|{longitude:.4f}|"
                        f"{kind}|zero".encode("utf-8")
                    ).hexdigest()
                    cached_nc = CACHE_DIR / f"{cache_key}.nc"

                    if cached_nc.exists() and cached_nc.stat().st_size > 1000:
                        shutil.copy2(cached_nc, dest)
                    else:
                        _create_zero_accum(dest, instant_path)
                        shutil.copy2(dest, cached_nc)

                    paths[kind] = str(dest.relative_to(BASE_DIR))
                    continue

                url = _query(
                    source, cycle, forecast_hour,
                    latitude, longitude, kind
                )
                cache_key = hashlib.sha256(
                    f"{CACHE_VERSION}|{url}".encode("utf-8")
                ).hexdigest()
                cached_grib = CACHE_DIR / f"{cache_key}.grib2"
                cached_nc = CACHE_DIR / f"{cache_key}.nc"

                if cached_nc.exists() and cached_nc.stat().st_size > 1000:
                    shutil.copy2(cached_nc, dest)
                else:
                    grib = temp_dir / f"{kind}.grib2"
                    if (
                        cached_grib.exists()
                        and cached_grib.stat().st_size > 10000
                        and cached_grib.open("rb").read(4) == b"GRIB"
                    ):
                        shutil.copy2(cached_grib, grib)
                    else:
                        cached_grib.unlink(missing_ok=True)
                        _download(url, grib)
                        shutil.copy2(grib, cached_grib)

                    _convert(grib, dest, kind)
                    shutil.copy2(dest, cached_nc)

                paths[kind] = str(dest.relative_to(BASE_DIR))
                if kind == "instant":
                    instant_path = dest

    except Exception:
        for relative_path in paths.values():
            (BASE_DIR / relative_path).unlink(missing_ok=True)
        raise

    return {
        "source": source,
        "cycle": cycle.isoformat(),
        "forecast_hour": forecast_hour,
        "paths": paths,
        "name": (
            f"{SOURCE_CONFIG[source]['label']} "
            f"{cycle:%Y-%m-%d %H}Z f{forecast_hour:03d}"
        ),
        "uploaded_by": uploaded_by,
    }
