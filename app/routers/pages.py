
from pathlib import Path
import json
import csv
import math
import shutil
import io

from datetime import datetime
from fastapi import APIRouter, Request, Form, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import current_user
from app.models import User, MeteorologyDataset, Simulation, SimulationResult
from app.schemas.simulation import SimulationConfig
from app.services.meteorology import MeteorologyService
from app.services.surface_meteorology import SurfaceMeteorologyService
from app.services.turbulence import TurbulenceService
from app.services.particle_simulation import ParticleSimulationService
from app.services.particle import ParticleService
from app.services.eruption import EruptionService
from app.services.deposition import DepositionService
from app.services.dispersion import DispersionService
from app.services.geojson import GeoJSONService
from app.services.concentration import ConcentrationService
from app.services.simulation import SimulationService
from app.services.mass_balance import MassBalanceService
from app.services.landcover import LandCoverService
from app.services.terrain_service import TerrainService
from app.services.noaa_source import acquire_noaa_dataset, SOURCE_CONFIG


router = APIRouter()


def format_significant(value, digits=4):
    if value is None or not isinstance(value, (int, float, str)):
        return "-"
    try:
        value = float(value)
        if not math.isfinite(value):
            return "-"
        if value == 0:
            return "0"
        return f"{value:.{digits}g}"
    except (ValueError, TypeError):
        return "-"


templates = Jinja2Templates(directory="app/templates")
templates.env.filters["sig"] = format_significant
BASE_DIR = Path(__file__).resolve().parents[2]

BACKEND_VEI = 3
BACKEND_PARTICLE_DENSITY = 2500.0
BACKEND_MER_METHOD = "degruyter"

terrain_service = TerrainService()

landcover_service = LandCoverService(
    data_dir=BASE_DIR / "data" / "landcover" / "ESA_WorldCover_2021"
)


def get_dataset_or_none(db, dataset_id, user):
    dataset = db.get(MeteorologyDataset, dataset_id)
    if not dataset or dataset.status != "valid" or not dataset.upper_air_path:
        return None
    return dataset



def get_simulation_service(
    dataset: MeteorologyDataset,
    config: SimulationConfig
):
    upper_air = BASE_DIR / dataset.upper_air_path
    surface_instant = (
        BASE_DIR / dataset.surface_instant_path
        if dataset.surface_instant_path else None
    )
    surface_accum = (
        BASE_DIR / dataset.surface_accum_path
        if dataset.surface_accum_path else None
    )

    meteorology = MeteorologyService(str(upper_air))
    surface_meteorology = None

    try:
        if surface_instant and surface_accum:
            surface_meteorology = SurfaceMeteorologyService(
                str(surface_instant),
                str(surface_accum)
            )

        turbulence = TurbulenceService(
            ventilation_rate=config.ventilation_rate,
            pbl_wind_speed=config.mean_pbl_wind,
            convective_velocity=config.convective_velocity,
            timestep_factor=config.timestep_factor
        )

        eruption = EruptionService(
            mer_method=config.mer_method
        )

        particle = ParticleService(
            total_particles=config.total_particles,
            particle_density=config.particle_density
        )

        particle_simulation = ParticleSimulationService(
            meteorology,
            turbulence
        )

        deposition = DepositionService(
            rain_a=config.scavenging_a,
            rain_b=config.scavenging_b
        )

        dispersion = DispersionService(
            particle_simulation=particle_simulation,
            meteorology=meteorology,
            deposition=deposition,
            turbulence=turbulence,
            surface_meteorology=surface_meteorology,
            landcover=landcover_service,
            terrain=terrain_service
        )

        dispersion.precipitation_source = config.precipitation_source
        dispersion.precipitation_rate = config.precipitation_rate

        concentration = ConcentrationService()
        geojson = GeoJSONService()

        return SimulationService(
            meteorology=meteorology,
            eruption=eruption,
            particle=particle,
            dispersion=dispersion,
            geojson=geojson,
            concentration=concentration,
            landcover=landcover_service,
            surface_meteorology=surface_meteorology,
            terrain=terrain_service
        )

    except Exception:
        meteorology.close()
        if surface_meteorology:
            surface_meteorology.close()
        raise



def dataset_grid_settings(meteorology, latitude, longitude):
    meteorology.load()

    latitudes = [
        float(x) for x in meteorology.data["latitude"]
    ]
    longitudes = [
        float(x) for x in meteorology.data["longitude"]
    ]

    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError(
            "Dataset meteorologi membutuhkan minimal dua titik "
            "latitude dan longitude."
        )

    source_lat = float(latitude)
    source_lon = float(longitude)

    if not all(math.isfinite(x) for x in (source_lat, source_lon)):
        raise ValueError("Koordinat gunung api tidak valid.")

    lat_min = min(latitudes)
    lat_max = max(latitudes)
    lon_min = min(longitudes)
    lon_max = max(longitudes)

    if not lat_min <= source_lat <= lat_max:
        raise ValueError(
            f"Latitude {source_lat} berada di luar cakupan "
            f"dataset ({lat_min} sampai {lat_max})."
        )

    if not lon_min <= source_lon <= lon_max:
        raise ValueError(
            f"Longitude {source_lon} berada di luar cakupan "
            f"dataset ({lon_min} sampai {lon_max})."
        )

    lat_step = abs(latitudes[1] - latitudes[0])
    lon_step = abs(longitudes[1] - longitudes[0])

    lat_resolution = lat_step * 111320.0
    cos_lat = max(
        abs(math.cos(math.radians(source_lat))),
        1e-8
    )
    lon_resolution = lon_step * 111320.0 * cos_lat

    grid_resolution = (
        lat_resolution + lon_resolution
    ) / 2.0

    north_distance = abs(source_lat - lat_max) * 111320.0
    south_distance = abs(source_lat - lat_min) * 111320.0
    east_distance = abs(source_lon - lon_max) * 111320.0 * cos_lat
    west_distance = abs(source_lon - lon_min) * 111320.0 * cos_lat

    domain_radius = min(
        north_distance,
        south_distance,
        east_distance,
        west_distance
    )

    if domain_radius <= 0:
        raise ValueError(
            "Koordinat berada pada batas dataset meteorologi. "
            "Diperlukan cakupan grid yang lebih luas."
        )

    return {
        "grid_resolution": float(grid_resolution),
        "domain_radius": float(domain_radius),
        "latitude_resolution": float(lat_resolution),
        "longitude_resolution": float(lon_resolution),
        "latitude_min": lat_min,
        "latitude_max": lat_max,
        "longitude_min": lon_min,
        "longitude_max": lon_max
    }


def build_result_display(result, config, dataset, info):
    eruption = result.get("eruption") or {}
    mass = result.get("mass_balance") or {}
    concentration = result.get("concentration") or {}
    deposition = result.get("surface_deposition") or {}
    spatial = result.get("spatial_analysis") or {}
    geojson = result.get("geojson") or {}

    mechanism_mass = (
        mass.get("deposition_mechanism_mass_kg") or {}
    )

    total_mass = float(
        eruption.get("total_mass") or 0.0
    )
    deposited = float(
        mass.get("deposited_mass_kg") or 0.0
    )
    airborne = float(
        mass.get("airborne_mass_kg") or 0.0
    )
    wet = float(
        mechanism_mass.get("wet") or 0.0
    )
    dry = float(
        mechanism_mass.get("dry") or 0.0
    )
    gravitational = float(
        mechanism_mass.get("gravitational") or 0.0
    )

    def percentage(value):
        return (
            round(float(value) / total_mass * 100, 2)
            if total_mass > 0 else 0.0
        )

    column_values = []
    time_series = concentration.get("time_series") or []

    if time_series:
        cells = time_series[-1].get("cells") or []
        columns = {}

        for cell in cells:
            key = (cell.get("i"), cell.get("j"))
            columns.setdefault(
                key,
                {
                    "mass": 0.0,
                    "latitude": float(cell["latitude"])
                }
            )
            columns[key]["mass"] += float(
                cell.get("mass") or 0.0
            )

        earth_radius = 6.371e6
        dlat = float(
            concentration.get("metadata", {}).get("dlat", 0.25)
        )
        dlon = float(
            concentration.get("metadata", {}).get("dlon", 0.25)
        )

        for column in columns.values():
            phi = math.radians(column["latitude"])
            area = (
                earth_radius ** 2 *
                abs(math.cos(phi)) *
                math.radians(dlat) *
                math.radians(dlon)
            )
            if area > 0:
                column_values.append(
                    column["mass"] / area
                )

    peak_loading = (
        max(column_values) if column_values else None
    )
    mean_loading = (
        sum(column_values) / len(column_values)
        if column_values else None
    )

    effective_timestep = float(
        result["config"]["timestep"]
    )
    turbulence_params = (
        result["config"].get("turbulence") or {}
    )

    result["location"] = {
        **(result.get("location") or {}),
        "latitude": config.latitude,
        "longitude": config.longitude,
        "altitude": config.altitude
    }

    result["input"] = {
        **(result.get("input") or {}),
        "volcano": result.get(
            "volcano_name", "Unknown Volcano"
        ),
        "latitude": config.latitude,
        "longitude": config.longitude,
        "altitude": config.altitude,
        "eruption_height": config.eruption_height,
        "eruption_duration": config.eruption_duration,
        "eruption_duration_hours": config.eruption_duration / 3600.0,
        "duration": config.duration,
        "duration_hours": config.duration / 3600.0,
        "timestep": effective_timestep,
        "timestep_source": "M6_LAGRANGIAN",
        "turbulence": turbulence_params,
        "output_interval_minutes": effective_timestep / 60.0,
        "vei": config.vei,
        "total_particles": config.total_particles,
        "particle_density": config.particle_density,
        "precipitation": config.precipitation,
        "precipitation_mm": config.precipitation * 1000.0,
        "grid_resolution": config.grid_resolution,
        "domain_radius": config.domain_radius,
        "mer_method": config.mer_method,
        "time_index": config.time_index,
        "meteorology_dataset": dataset.name
    }

    result["total_erupted_mass"] = total_mass
    result["mer"] = float(
        eruption.get("mer") or 0.0
    )
    result["output_interval_minutes"] = round(
        effective_timestep / 60.0, 2
    )

    result["mass_partition"] = {
        "airborne": percentage(airborne),
        "wet_deposition": percentage(wet),
        "dry_deposition": percentage(dry),
        "gravitational": percentage(gravitational)
    }

    result["mass_closure"] = mass.get(
        "closure_percent"
    )
    result["peak_column_loading"] = peak_loading
    result["mean_column_loading"] = mean_loading

    result["spatial_analysis"] = spatial
    result["plume_length"] = spatial.get(
        "plume_length_km"
    )
    result["plume_width"] = spatial.get(
        "plume_width_km"
    )
    result["affected_area"] = spatial.get(
        "footprint_km2"
    )
    result["mean_wind_speed"] = eruption.get(
        "mean_wind_speed"
    )

    result["geojson"] = geojson
    result["concentration"] = concentration
    result["surface_deposition"] = deposition

    simulation_id = result.get("simulation_id")

    result["export_pdf_url"] = (
        f"/reports/{simulation_id}/pdf"
    )
    result["export_kmz_url"] = (
        f"/reports/{simulation_id}/kmz"
    )
    result["export_csv_url"] = (
        f"/reports/{simulation_id}/csv"
    )

    return result


@router.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    user: User = Depends(current_user)
):
    if not user:
        return RedirectResponse("/login", 303)
    return RedirectResponse("/dashboard", 303)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login", 303)

    simulations = db.query(Simulation).filter(
        Simulation.user_id == user.id
    ).order_by(
        Simulation.created_at.desc()
    ).all()

    datasets = db.query(MeteorologyDataset).order_by(
        MeteorologyDataset.created_at.desc()
    ).all()

    total_simulations = len(simulations)
    processing_simulations = sum(
        1 for s in simulations if s.status == "running"
    )
    completed_simulations = sum(
        1 for s in simulations if s.status == "completed"
    )
    failed_simulations = sum(
        1 for s in simulations if s.status == "failed"
    )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request": request,
            "user": user,
            "simulations": simulations,
            "datasets": datasets,
            "total_simulations": total_simulations,
            "processing_simulations": processing_simulations,
            "completed_simulations": completed_simulations,
            "failed_simulations": failed_simulations
        }
    )


@router.get(
    "/simulations/{simulation_id}/result",
    response_class=HTMLResponse
)
async def simulation_history_result(
    simulation_id: int,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login", 303)

    simulation = db.get(Simulation, simulation_id)

    if not simulation or simulation.user_id != user.id:
        return RedirectResponse(
            "/dashboard?error=not_found", 303
        )

    return templates.TemplateResponse(
        request=request,
        name="simulations/history_result.html",
        context={
            "request": request,
            "user": user,
            "simulation": simulation,
            "result": simulation.result
        }
    )


@router.post("/simulations/{simulation_id}/delete")
async def delete_simulation(
    simulation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user)
):
    if not user:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "message": "Session login berakhir. Silakan login kembali."
            }
        )

    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == user.id
    ).first()

    if not simulation:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Simulation tidak ditemukan."
            }
        )

    try:
        db.delete(simulation)
        db.commit()
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Simulation berhasil dihapus."
            }
        )
    except Exception as exc:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(exc)
            }
        )


@router.get("/simulations/create")
async def create_simulation(
    request: Request,
    user: User = Depends(current_user)
):
    if not user:
        return RedirectResponse("/login", 303)

    params = request.query_params

    query = {
        key: params.get(key)
        for key in (
            "volcano", "latitude", "longitude", "altitude"
        )
        if params.get(key) not in (None, "")
    }

    from urllib.parse import urlencode

    url = "/meteorology/config"
    if query:
        url += "?" + urlencode(query)

    return RedirectResponse(url, 303)



@router.get("/simulations/meteorology/{dataset_id}")
async def simulation_meteorology(
    dataset_id: int,
    request: Request,
    latitude: float = -7.5407,
    longitude: float = 110.4462,
    time_index: int = 0,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return JSONResponse(
            {"detail": "Unauthorized"},
            status_code=401
        )

    dataset = get_dataset_or_none(
        db, dataset_id, user
    )

    if not dataset:
        return JSONResponse(
            {"detail": "Meteorology dataset tidak valid."},
            status_code=404
        )

    meteorology = None
    surface = None

    try:
        meteorology = MeteorologyService(
            str(BASE_DIR / dataset.upper_air_path)
        )

        info = meteorology.time_information()
        times = info.get("times", [])

        if not times:
            raise ValueError(
                "Dataset Upper-Air tidak memiliki waktu valid."
            )

        if time_index < 0 or time_index >= len(times):
            raise ValueError(
                f"Time index di luar rentang dataset. "
                f"Jumlah waktu tersedia: {len(times)}."
            )

        grid = dataset_grid_settings(
            meteorology, latitude, longitude
        )

        upper_air = meteorology.atmospheric_profile(
            latitude, longitude, time_index
        )

        profile = []

        for level in upper_air["profile"]:
            u = float(level["u"])
            v = float(level["v"])

            profile.append({
                **level,
                "wind_speed": math.hypot(u, v)
            })

        upper_air["profile"] = profile

        upper_air["latitude"] = float(latitude)
        upper_air["longitude"] = float(longitude)
        upper_air["time_index"] = int(time_index)
        upper_air["time"] = str(times[time_index])
        upper_air["level_count"] = len(profile)

        surface_profile = None
        surface_info = None

        if dataset.surface_instant_path and dataset.surface_accum_path:
            surface = SurfaceMeteorologyService(
                str(BASE_DIR / dataset.surface_instant_path),
                str(BASE_DIR / dataset.surface_accum_path)
            )

            surface_data = surface.load()

            if time_index >= len(surface_data["times"]):
                raise ValueError(
                    "Time index Surface tidak tersedia pada dataset."
                )

            surface_profile = surface.surface_profile(
                latitude, longitude, time_index
            )

            surface_info = {
                "time_count": len(surface_data["times"]),
                "precipitation_units": surface_data["precipitation_units"],
                "has_precipitation": True,
                "has_surface_pressure": True,
                "has_friction_velocity": True,
                "has_temperature": True,
                "has_heat_flux": True
            }

        turbulence_parameters = TurbulenceService().parameters()

        pbl = {
            "ventilation_rate": turbulence_parameters.get(
                "ventilation_rate"
            ),
            "mean_pbl_wind": turbulence_parameters.get(
                "pbl_wind_speed"
            ),
            "convective_velocity": turbulence_parameters.get(
                "convective_velocity"
            ),
            "mixing_height": turbulence_parameters.get(
                "mixing_height"
            ),
            "tau_l": turbulence_parameters.get(
                "tau_l"
            ),
            "timestep": turbulence_parameters.get(
                "timestep"
            )
        }

        return JSONResponse({
            "dataset_id": dataset.id,
            "name": dataset.name,
            "times": [str(x) for x in times],
            "time_count": info.get(
                "time_count",
                info.get("count", len(times))
            ),
            "interval_seconds": info.get("interval_seconds"),
            "interval_hours": info.get("interval_hours"),
            "elapsed_seconds": info.get("elapsed_seconds", []),
            "grid_resolution": grid["grid_resolution"],
            "domain_radius": grid["domain_radius"],
            "latitude_resolution": grid["latitude_resolution"],
            "longitude_resolution": grid["longitude_resolution"],
            "upper_air": upper_air,
            "surface_info": surface_info,
            "surface": surface_profile,
            "pbl": pbl
        })

    except Exception as exc:
        return JSONResponse(
            {"detail": str(exc)},
            status_code=400
        )

    finally:
        if meteorology:
            meteorology.close()

        if surface:
            surface.close()


@router.get("/api/terrain/info")
async def terrain_info(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    user: User = Depends(current_user)
):
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    result = {
        "latitude": latitude,
        "longitude": longitude,
        "terrain": None,
        "landcover": None,
        "errors": {}
    }

    try:
        result["terrain"] = terrain_service.elevation_at(
            latitude, longitude
        )
    except Exception as exc:
        result["errors"]["terrain"] = str(exc)

    try:
        result["landcover"] = landcover_service.surface_at(
            latitude, longitude
        )
    except Exception as exc:
        result["errors"]["landcover"] = str(exc)

    result["status"] = (
        "available"
        if result["terrain"] and result["landcover"]
        else "partial"
        if result["terrain"] or result["landcover"]
        else "unavailable"
    )

    return JSONResponse(result)


@router.get("/api/terrain/status")
async def terrain_status(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    user: User = Depends(current_user)
):
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    try:
        return JSONResponse({
            "terrain": terrain_service.get_tile_status(
                latitude, longitude
            ),
            "landcover_tile": landcover_service._tile_name(
                latitude, longitude
            )
        })
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


@router.post("/simulations/run", response_class=HTMLResponse)
async def run_simulation(
    request: Request,
    meteorology_dataset_id: int = Form(...),
    name: str = Form("Volcanic Ash Simulation"),
    volcano: str = Form("Merapi"),
    latitude: float = Form(...),
    longitude: float = Form(...),
    altitude: float = Form(...),
    eruption_height: float = Form(...),
    eruption_duration: float = Form(...),
    duration: float = Form(...),
    time_index: int = Form(0),
    total_particles: int = Form(50000),
    precipitation_source: str = Form("dataset"),
    precipitation_rate: float | None = Form(None),
    scavenging_a: float = Form(8.4e-5),
    scavenging_b: float = Form(0.79),
    ventilation_rate: float = Form(4000.0),
    mean_pbl_wind: float = Form(7.018),
    convective_velocity: float = Form(2.0),
    timestep_factor: float = Form(1.0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):

    if not user:
        return RedirectResponse("/login", 303)

    dataset = get_dataset_or_none(
        db, meteorology_dataset_id, user
    )

    if not dataset:
        datasets = db.query(MeteorologyDataset).filter(
            MeteorologyDataset.status == "valid",
            MeteorologyDataset.upper_air_path.is_not(None)
        ).order_by(
            MeteorologyDataset.created_at.desc()
        ).all()

        return templates.TemplateResponse(
            request=request,
            name="simulations/create.html",
            context={
                "request": request,
                "user": user,
                "datasets": datasets,
                "meteorology": None,
                "error": (
                    "Meteorology dataset tidak valid "
                    "atau Upper-Air belum tersedia."
                )
            },
            status_code=400
        )

    service = None
    meteorology = None
    surface = None
    simulation = None

    try:
        upper_air = BASE_DIR / dataset.upper_air_path
        meteorology = MeteorologyService(
            str(upper_air)
        )

        info = meteorology.time_information()

        if time_index < 0 or time_index >= len(info["times"]):
            raise ValueError(
                f"Time index {time_index} berada di luar range dataset. "
                f"Jumlah waktu tersedia: {len(info['times'])}."
            )

        grid = dataset_grid_settings(
            meteorology,
            latitude,
            longitude
        )

        if (
            not dataset.surface_instant_path
            or not dataset.surface_accum_path
        ):
            raise ValueError(
                "Dataset belum memiliki Surface Instant "
                "dan Surface Accumulated."
            )

        surface = SurfaceMeteorologyService(
            str(BASE_DIR / dataset.surface_instant_path),
            str(BASE_DIR / dataset.surface_accum_path)
        )

        surface_profile = surface.surface_profile(
            latitude,
            longitude,
            time_index
        )

        
        if precipitation_source not in (
            "dataset",
            "manual",
            "none"
        ):
            raise ValueError(
                "Sumber presipitasi tidak valid."
            )

        if precipitation_source == "dataset":
            if not surface_profile.get("precipitation_available"):
                raise ValueError(
                    "Data presipitasi GFS tidak tersedia."
                )

            precipitation = float(
                surface_profile["precipitation"]
            )

            if (
                not math.isfinite(precipitation)
                or precipitation < 0
            ):
                raise ValueError(
                    "Presipitasi dataset tidak valid."
                )

        elif precipitation_source == "manual":
            if precipitation_rate is None:
                raise ValueError(
                    "Laju hujan manual wajib diisi."
                )

            if (
                not math.isfinite(precipitation_rate)
                or precipitation_rate < 0
            ):
                raise ValueError(
                    "Laju hujan manual harus >= 0 mm/jam."
                )

            precipitation = 0.0

        else:
            precipitation = 0.0


        precipitation = float(
            surface_profile["precipitation"]
        )

        if not math.isfinite(precipitation) or precipitation < 0:
            raise ValueError(
                "Nilai presipitasi tidak valid."
            )

        meteorology.close()
        meteorology = None

        surface.close()
        surface = None

        
        config = SimulationConfig(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            eruption_height=eruption_height,
            eruption_duration=eruption_duration * 3600.0,
            duration=duration * 3600.0,
            time_index=time_index,
            vei=BACKEND_VEI,
            total_particles=total_particles,
            particle_density=BACKEND_PARTICLE_DENSITY,
            grid_resolution=grid["grid_resolution"],
            domain_radius=grid["domain_radius"],
            precipitation=precipitation,
            mer_method=BACKEND_MER_METHOD,
            precipitation_source=precipitation_source,
            precipitation_rate=precipitation_rate,
            scavenging_a=scavenging_a,
            scavenging_b=scavenging_b,
            ventilation_rate=ventilation_rate,
            mean_pbl_wind=mean_pbl_wind,
            convective_velocity=convective_velocity,
            timestep_factor=timestep_factor
        )

        turbulence_params = TurbulenceService(
            ventilation_rate=config.ventilation_rate,
            pbl_wind_speed=config.mean_pbl_wind,
            convective_velocity=config.convective_velocity,
            timestep_factor=config.timestep_factor
        ).parameters()

        effective_dt = float(turbulence_params["timestep"])

        print("\n========== INPUT SIMULATION ==========")
        print("Volcano          :", volcano)
        print("Latitude         :", config.latitude)
        print("Longitude        :", config.longitude)
        print("Vent altitude    :", config.altitude)
        print("Eruption height  :", config.eruption_height)
        print("Eruption duration:", config.eruption_duration)
        print("Simulation time  :", config.duration)
        print("Timestep M6      :", effective_dt)
        print("Particles        :", config.total_particles)
        print("Time index       :", config.time_index)
        print("Rain source      :", config.precipitation_source)
        print("Rain rate        :", config.precipitation_rate)
        print("Scavenging A     :", config.scavenging_a)
        print("Scavenging B     :", config.scavenging_b)
        print("Ventilation rate :", config.ventilation_rate)
        print("Mean PBL wind    :", config.mean_pbl_wind)
        print("Convective vel.  :", config.convective_velocity)
        print("Timestep factor  :", config.timestep_factor)
        print("======================================\n")


        simulation = Simulation(
            user_id=user.id,
            meteorology_dataset_id=dataset.id,
            name=name,
            latitude=config.latitude,
            longitude=config.longitude,
            altitude=config.altitude,
            eruption_height=config.eruption_height,
            eruption_duration=config.eruption_duration,
            duration=config.duration,
            timestep=effective_dt,
            time_index=config.time_index,
            precipitation=config.precipitation,
            total_particles=config.total_particles,
            particle_density=config.particle_density,
            grid_resolution=config.grid_resolution,
            domain_radius=config.domain_radius,
            mer_method=config.mer_method,
            vei=config.vei,
            status="running"
        )

        db.add(simulation)
        db.commit()
        db.refresh(simulation)

        service = get_simulation_service(
            dataset=dataset,
            config=config
        )

        info = service.meteorology.time_information()

        result = service.run(config)

        actual_dt = float(
            result["config"]["timestep"]
        )

        simulation.timestep = actual_dt

        mass = result.get("mass_balance") or {}
        deposition = result.get("surface_deposition") or {}

        source_mass = float(
            mass.get("source_mass_kg", 0.0)
        )
        deposited_mass = float(
            mass.get("deposited_mass_kg", 0.0)
        )
        airborne_mass = float(
            mass.get("airborne_mass_kg", 0.0)
        )
        grid_deposited_mass = float(
            deposition.get("total_deposited_mass", 0.0)
        )

        validation = MassBalanceService().validate(
            source_mass=source_mass,
            deposited_mass=deposited_mass,
            airborne_mass=airborne_mass,
            grid_deposited_mass=grid_deposited_mass
        )

        print("\n========== MASS BALANCE TEST ==========")
        print(f"Source mass          : {source_mass:.6f} kg")
        print(f"Deposited mass       : {deposited_mass:.6f} kg")
        print(f"Airborne mass        : {airborne_mass:.6f} kg")
        print(f"Simulated mass       : {validation['simulated_mass']:.6f} kg")
        print(f"Mass error           : {validation['error']:.12f} kg")
        print(f"Relative error       : {validation['relative_error']:.12e}")
        print(f"Tolerance            : {validation['tolerance']:.12e} kg")
        print(f"Grid deposited mass  : {grid_deposited_mass:.6f} kg")
        print(f"Grid mass error      : {validation['grid_mass_error']:.12f} kg")
        print(f"Mass balance valid   : {validation['valid']}")
        print(f"Grid balance valid   : {validation['grid_valid']}")
        print(f"Overall valid        : {validation['valid_all']}")
        print("=======================================\n")

        mass["validation"] = validation
        mass["valid"] = validation["valid"]
        mass["grid_valid"] = validation["grid_valid"]
        mass["valid_all"] = validation["valid_all"]
        mass["error"] = validation["error"]
        mass["grid_mass_error"] = validation["grid_mass_error"]

        result["simulation_id"] = simulation.id
        result["volcano_name"] = (
            volcano or "Unknown Volcano"
        )
        result["vei"] = config.vei
        result["simulation_time"] = str(
            info["times"][config.time_index]
        )

        result["input"] = {
            "volcano": volcano or "Unknown Volcano",
            "latitude": config.latitude,
            "longitude": config.longitude,
            "altitude": config.altitude,
            "eruption_height": config.eruption_height,
            "eruption_duration": config.eruption_duration,
            "eruption_duration_hours": (
                config.eruption_duration / 3600.0
            ),
            "vei": config.vei,
            "duration": config.duration,
            "duration_hours": config.duration / 3600.0,
            "timestep": actual_dt,
            "timestep_source": "M6_LAGRANGIAN",
            "precipitation": config.precipitation,
            "precipitation_m": config.precipitation,
            "precipitation_mm": config.precipitation * 1000.0,
            "precipitation_source": (
                f"{dataset.name} — surface accumulated"
            ),
            "total_particles": config.total_particles,
            "particle_density": config.particle_density,
            "grid_resolution": config.grid_resolution,
            "domain_radius": config.domain_radius,
            "grid_resolution_source": (
                f"{dataset.name} — meteorological grid"
            ),
            "domain_radius_source": (
                f"{dataset.name} — dataset spatial extent"
            ),
            "mer_method": config.mer_method,
            "time_index": config.time_index,
            "meteorology_time": str(
                info["times"][config.time_index]
            ),
            "meteorology_dataset_id": dataset.id,
            "meteorology_dataset": dataset.name
        }

        result = build_result_display(
            result, config, dataset, info
        )

        output_dir = (
            BASE_DIR / "data" / "simulations" / str(simulation.id)
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        geojson = result.get("geojson", {})

        files = {
            "trajectory_path": (
                "trajectory.geojson",
                geojson.get("trajectory")
            ),
            "particles_path": (
                "particles.geojson",
                geojson.get("points")
            ),
            "deposition_path": (
                "deposition.geojson",
                geojson.get("deposition")
            ),
            "deposition_grid_path": (
                "deposition_grid.geojson",
                geojson.get("deposition_grid")
            )
        }

        saved_paths = {}

        for key, (filename, data) in files.items():
            if data is None:
                continue

            path = output_dir / filename
            path.write_text(
                json.dumps(data, default=str),
                encoding="utf-8"
            )
            saved_paths[key] = str(
                path.relative_to(BASE_DIR)
            )

        eruption = result.get("eruption") or {}
        mass = result.get("mass_balance") or {}
        deposition = result.get("surface_deposition") or {}

        simulation_result = SimulationResult(
            simulation_id=simulation.id,
            total_mass=eruption.get("total_mass"),
            mer=eruption.get("mer"),
            deposited_mass=mass.get("deposited_mass_kg"),
            airborne_mass=mass.get("airborne_mass_kg"),
            grid_deposited_mass=deposition.get(
                "total_deposited_mass"
            ),
            mass_error=mass.get("error"),
            validation_status=mass.get("valid_all"),
            trajectory_path=saved_paths.get(
                "trajectory_path"
            ),
            particles_path=saved_paths.get(
                "particles_path"
            ),
            deposition_path=saved_paths.get(
                "deposition_path"
            ),
            deposition_grid_path=saved_paths.get(
                "deposition_grid_path"
            )
        )

        db.add(simulation_result)

        simulation.status = "completed"
        simulation.completed_at = datetime.utcnow()

        db.commit()

    except Exception as exc:
        db.rollback()

        if simulation is not None and simulation.id is not None:
            simulation.status = "failed"
            simulation.error_message = str(exc)
            db.commit()

        raise

    finally:
        if service:
            if getattr(service, "meteorology", None):
                service.meteorology.close()

            if getattr(service, "surface_meteorology", None):
                service.surface_meteorology.close()

        if meteorology:
            meteorology.close()

        if surface:
            surface.close()

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={
            "request": request,
            "user": user,
            "result": result,
            "simulation": simulation
        }
    )


@router.get("/meteorology/config", response_class=HTMLResponse)
async def meteorology_config(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login", 303)

    datasets = db.query(MeteorologyDataset).filter(
        MeteorologyDataset.status == "valid"
    ).order_by(
        MeteorologyDataset.created_at.desc()
    ).limit(20).all()

    return templates.TemplateResponse(
        request=request,
        name="meteorology/config.html",
        context={
            "request": request,
            "user": user,
            "datasets": datasets,
            "sources": SOURCE_CONFIG,
            "error": None,
            "message": None,
            "volcano": request.query_params.get("volcano", ""),
            "latitude": request.query_params.get("latitude", ""),
            "longitude": request.query_params.get("longitude", ""),
            "altitude": request.query_params.get("altitude", "")
        }
    )


@router.post("/meteorology/acquire", response_class=HTMLResponse)
async def meteorology_acquire(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login", 303)

    try:
        form = await request.form()

        source = form.get("source")
        valid_date = form.get("valid_date")
        forecast_hour = int(
            form.get("forecast_hour") or 0
        )
        latitude = float(
            form.get("latitude")
        )
        longitude = float(
            form.get("longitude")
        )
        duration_hours = int(
            form.get("duration_hours") or 12
        )

        if duration_hours <= 0 or duration_hours > 72:
            raise ValueError(
                "Durasi meteorologi harus 1–72 jam."
            )

        acquired = acquire_noaa_dataset(
            source,
            valid_date,
            forecast_hour,
            latitude,
            longitude,
            user.id,
            duration_hours=duration_hours
        )

        from app.services.netcdf_upload import inspect_netcdf

        upper = BASE_DIR / acquired["paths"]["upper"]
        instant = BASE_DIR / acquired["paths"]["instant"]
        accum = BASE_DIR / acquired["paths"]["accum"]

        upper_meta = inspect_netcdf(
            upper, "upper_air"
        )
        instant_meta = inspect_netcdf(
            instant, "surface"
        )
        accum_meta = inspect_netcdf(
            accum, "surface"
        )

        dataset = MeteorologyDataset(
            name=acquired["name"],
            description=(
                f"Sumber NOAA {acquired['source']}; "
                f"cycle {acquired['cycle']}; "
                f"forecast f{forecast_hour:03d}–"
                f"f{acquired['forecast_end_hour']:03d}; "
                f"durasi {duration_hours} jam."
            ),
            upper_air_path=acquired["paths"]["upper"],
            surface_instant_path=acquired["paths"]["instant"],
            surface_accum_path=acquired["paths"]["accum"],
            start_time=upper_meta.get("start_time"),
            end_time=upper_meta.get("end_time"),
            time_count=upper_meta.get("time_count"),
            interval_hours=upper_meta.get("interval_hours"),
            variables=json.dumps(sorted(set(
                upper_meta.get("variables", []) +
                instant_meta.get("variables", []) +
                accum_meta.get("variables", [])
            ))),
            status="valid",
            uploaded_by=user.id
        )

        db.add(dataset)
        db.commit()

        return RedirectResponse(
            f"/meteorology/config?downloaded=1"
            f"&volcano={form.get('volcano', '')}"
            f"&latitude={latitude}"
            f"&longitude={longitude}"
            f"&altitude={form.get('altitude', '')}",
            303
        )

    except Exception as exc:
        db.rollback()

        datasets = db.query(MeteorologyDataset).filter(
            MeteorologyDataset.status == "valid"
        ).order_by(
            MeteorologyDataset.created_at.desc()
        ).limit(20).all()

        return templates.TemplateResponse(
            request=request,
            name="meteorology/config.html",
            context={
                "request": request,
                "user": user,
                "datasets": datasets,
                "sources": SOURCE_CONFIG,
                "error": str(exc),
                "message": None,
                "volcano": (
                    form.get("volcano", "")
                    if "form" in locals() else ""
                ),
                "latitude": (
                    form.get("latitude", "")
                    if "form" in locals() else ""
                ),
                "longitude": (
                    form.get("longitude", "")
                    if "form" in locals() else ""
                ),
                "altitude": (
                    form.get("altitude", "")
                    if "form" in locals() else ""
                )
            },
            status_code=400
        )


@router.get("/source-config", response_class=HTMLResponse)
async def source_config(
    request: Request,
    user: User = Depends(current_user)
):
    if not user:
        return RedirectResponse("/login", 303)

    return templates.TemplateResponse(
        request=request,
        name="simulations/source_config.html",
        context={
            "request": request,
            "user": user
        }
    )


@router.get("/dispersion", response_class=HTMLResponse)
async def dispersion(
    request: Request,
    user: User = Depends(current_user)
):
    if not user:
        return RedirectResponse("/login", 303)

    params = request.query_params

    q = {
        key: params.get(key)
        for key in (
            "meteorology_dataset_id",
            "volcano",
            "latitude",
            "longitude",
            "altitude",
            "name",
            "eruption_height",
            "eruption_duration",
            "time_index",
            "total_particles"
        )
        if params.get(key) not in (None, "")
    }

    return templates.TemplateResponse(
        request=request,
        name="simulations/dispersion.html",
        context={
            "request": request,
            "user": user,
            "q": q
        }
    )


@router.get("/reports", response_class=HTMLResponse)
async def reports(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login", 303)

    simulations = db.query(Simulation).filter(
        Simulation.user_id == user.id
    ).order_by(
        Simulation.created_at.desc()
    ).all()

    return templates.TemplateResponse(
        request=request,
        name="reports/index.html",
        context={
            "request": request,
            "user": user,
            "simulations": simulations
        }
    )


@router.get("/reports/{simulation_id}/csv")
async def report_csv(
    simulation_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login", 303)

    simulation = db.get(
        Simulation, simulation_id
    )

    if not simulation or simulation.user_id != user.id:
        return JSONResponse(
            {"detail": "Laporan tidak ditemukan."},
            status_code=404
        )

    result = simulation.result
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["field", "value"])

    for key, value in [
        ("simulation_id", simulation.id),
        ("name", simulation.name),
        ("status", simulation.status),
        ("created_at", simulation.created_at),
        ("latitude", simulation.latitude),
        ("longitude", simulation.longitude),
        ("altitude_m", simulation.altitude),
        ("eruption_height_m", simulation.eruption_height),
        ("eruption_duration_s", simulation.eruption_duration),
        ("duration_s", simulation.duration),
        ("timestep_s", simulation.timestep),
        ("precipitation", simulation.precipitation),
        ("particles", simulation.total_particles),
        ("grid_resolution_m", simulation.grid_resolution),
        ("domain_radius_m", simulation.domain_radius)
    ]:
        writer.writerow([key, value])

    if result:
        for key, value in [
            ("MER_kg_s", result.mer),
            ("total_mass_kg", result.total_mass),
            ("deposited_mass_kg", result.deposited_mass),
            ("airborne_mass_kg", result.airborne_mass),
            ("grid_deposited_mass_kg", result.grid_deposited_mass),
            ("mass_error", result.mass_error),
            ("mass_balance_valid", result.validation_status)
        ]:
            writer.writerow([key, value])

    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment; filename="
                f"simulation-{simulation.id}-report.csv"
            )
        }
    )
