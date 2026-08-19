from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.schemas.simulation import SimulationConfig
from app.services.meteorology import MeteorologyService
from app.services.turbulence import TurbulenceService
from app.services.particle import ParticleService
from app.services.particle_simulation import ParticleSimulationService
from app.services.dispersion import DispersionService
from app.services.eruption import EruptionService
from app.services.geojson import GeoJSONService

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
BASE_DIR = Path(__file__).resolve().parents[2]

meteorology_service = MeteorologyService(str(BASE_DIR / "data" / "era5_pressure_jan_2026.nc"))
turbulence = TurbulenceService()
particle_service = ParticleService()
eruption_service = EruptionService()

particle_simulation = ParticleSimulationService(
    meteorology=meteorology_service,
    turbulence=turbulence,
    particle=particle_service
)

dispersion = DispersionService(
    particle_simulation=particle_simulation,
    meteorology=meteorology_service
)

geojson_service = GeoJSONService()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={})


@router.get("/simulations/create", response_class=HTMLResponse)
async def create_simulation(request: Request):
    time_information = meteorology_service.time_information()
    return templates.TemplateResponse(request=request, name="simulations/create.html", context={"meteorology": time_information})


@router.post("/simulations/run", response_class=HTMLResponse)
async def run_simulation(
    request: Request,
    volcano_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    start_date: str = Form(...),
    start_time: str = Form(...),
    dataset: str = Form(...),
    time_index: int = Form(...),
    altitude: float = Form(...),
    eruption_height: float = Form(...),
    eruption_duration: float = Form(...),
    particle_count: int = Form(...),
    mean_radius: float = Form(...),
    sigma: float = Form(...),
    duration: int = Form(...),
    timestep: int = Form(...)
):
    config = SimulationConfig(
        volcano_name=volcano_name,
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        start_time=start_time,
        dataset=dataset,
        time_index=time_index,
        altitude=altitude,
        eruption_height=eruption_height,
        eruption_duration=eruption_duration,
        particle_count=particle_count,
        mean_radius=mean_radius,
        sigma=sigma,
        duration=duration,
        timestep=timestep
    )

    time_information = meteorology_service.time_information()
    meteorology_times = time_information.get("times", [])

    if config.time_index < 0 or config.time_index >= len(meteorology_times):
        raise ValueError(f"Invalid meteorological time index: {config.time_index}")

    atmospheric = meteorology_service.atmospheric_profile(
        latitude=config.latitude,
        longitude=config.longitude,
        time_index=config.time_index
    )

    eruption = eruption_service.calculate(
        atmospheric_profile=atmospheric,
        eruption_height=config.eruption_height,
        eruption_duration=config.eruption_duration
    )

    particle_summary = particle_service.create_particle_summary(
        total_mass=eruption["total_mass"]
    )

    particles = particle_service.create_particles(
        total_mass=eruption["total_mass"],
        latitude=config.latitude,
        longitude=config.longitude,
        altitude=config.altitude
    )

    grid = meteorology_service.nearest_grid(
        latitude=config.latitude,
        longitude=config.longitude
    )

    dispersion_result = dispersion.simulate(
        particles=particles,
        duration=config.duration,
        dt=config.timestep,
        initial_time_index=config.time_index
    )

    trajectory_geojson = geojson_service.trajectory_to_feature_collection(dispersion_result)
    point_geojson = geojson_service.trajectory_points_to_feature_collection(dispersion_result)

    total_particles = dispersion_result["total_particles"]

    simulation = {
        "config": {
            "volcano_name": config.volcano_name,
            "latitude": config.latitude,
            "longitude": config.longitude,
            "start_date": config.start_date,
            "start_time": config.start_time,
            "dataset": config.dataset,
            "time_index": config.time_index,
            "altitude": config.altitude,
            "eruption_height": config.eruption_height,
            "eruption_duration": config.eruption_duration,
            "particle_count": config.particle_count,
            "mean_radius": config.mean_radius,
            "sigma": config.sigma,
            "duration": config.duration,
            "timestep": config.timestep
        },
        "volcano_name": config.volcano_name,
        "latitude": config.latitude,
        "longitude": config.longitude,
        "altitude": config.altitude,
        "start_date": config.start_date,
        "start_time": config.start_time,
        "dataset": config.dataset,
        "time_index": config.time_index,
        "meteorology_time": str(meteorology_times[config.time_index]),
        "duration": config.duration,
        "dt": config.timestep,
        "grid": grid,
        "meteorology": dispersion_result["meteorology"],
        "atmospheric": atmospheric,
        "eruption": eruption,
        "particle_summary": particle_summary,
        "particles": {
            "total": total_particles,
            "groups": particles
        },
        "particle_count": dispersion_result["particle_count"],
        "total_particles": total_particles,
        "steps": dispersion_result["steps"],
        "trajectories": dispersion_result["trajectories"],
        "geojson": {
            "trajectory": trajectory_geojson,
            "points": point_geojson
        }
    }

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={"simulation": simulation}
    )


@router.get(
    "/simulations/result",
    response_class=HTMLResponse
)
async def simulation_result(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={
            "simulation": None
        }
    )


@router.get(
    "/simulations/result",
    response_class=HTMLResponse
)
async def simulation_result(
    request: Request
):

    return templates.TemplateResponse(

        request=request,

        name="simulations/result.html",

        context={}
    )


@router.get(
    "/simulations/result",
    response_class=HTMLResponse
)
async def simulation_result(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={}
    )

@router.get(
    "/simulations/result",
    response_class=HTMLResponse
)
async def simulation_result(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={}
    )


@router.get(
    "/simulations/result",
    response_class=HTMLResponse
)
async def simulation_result(
    request: Request
):
    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={}
    )


@router.get(
    "/simulations/result",
    response_class=HTMLResponse
)
async def simulation_result(
    request: Request
):
    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={
            "simulation": None
        }
    )

@router.get("/simulations/history", response_class=HTMLResponse)
async def simulation_history(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="simulations/history.html",
        context={}
    )

@router.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="reports/index.html",
        context={}
    )

@router.get("/api/meteorology")
async def meteorology():
    service = MeteorologyService("data/era_merapi.nc")
    data = service.load()

    return {
        "variables": data["variables"],
        "time_count": len(data["time"]),
        "pressure_level": data["pressure_level"],
        "latitude": data["latitude"],
        "longitude": data["longitude"]
    }


@router.get("/api/meteorology/profile")
async def meteorology_profile():
    service = MeteorologyService("data/era_merapi_jan_2026.nc")

    return service.atmospheric_profile(
        latitude=-7.54,
        longitude=110.44,
        time_index=config.time_index
    )

@router.get("/api/meteorology/stability")
async def meteorology_stability():
    meteorology = MeteorologyService(
        "data/era_merapi_jan_2026.nc"
    )

    atmosphere = meteorology.atmospheric_profile(
        latitude=-7.54,
        longitude=110.44,
        time_index=config.time_index
    )

    profile = atmosphere["profile"]

    altitude = [item["altitude"] for item in profile]
    theta = [item["potential_temperature"] for item in profile]

    turbulence = TurbulenceService()

    stability = turbulence.calculate_stability(
        potential_temperature=theta,
        altitude=altitude
    )

    return {
        "latitude": atmosphere["latitude"],
        "longitude": atmosphere["longitude"],
        "time_index": atmosphere["time_index"],
        "altitude": altitude,
        "potential_temperature": theta,
        "dtheta_dz": stability["dtheta_dz"].tolist(),
        "n_squared": stability["n_squared"].tolist(),
        "n": stability["n"].tolist()
    }

@router.get("/api/meteorology/coriolis")
async def meteorology_coriolis():
    turbulence = TurbulenceService()

    latitude = -7.54

    f = turbulence.calculate_coriolis_parameter(latitude)

    return {
        "latitude": latitude,
        "coriolis_parameter": f
    }