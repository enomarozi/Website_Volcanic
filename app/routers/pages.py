from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
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

meteorology_service = MeteorologyService(
    str(BASE_DIR / "data" / "era5_pressure_jan_2026.nc")
)

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
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={}
    )

@router.get("/simulations/create", response_class=HTMLResponse)
async def create_simulation(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="simulations/create.html",
        context={}
    )


@router.post(
    "/simulations/run",
    response_class=HTMLResponse
)
async def run_simulation(
    request: Request,
    latitude: float = Form(...),
    longitude: float = Form(...),
    altitude: float = Form(...),
    eruption_height: float = Form(...),
    eruption_duration: float = Form(...),
    duration: int = Form(...),
    dt: int = Form(...)
):
    atmospheric = (
        meteorology_service
        .atmospheric_profile(
            latitude=latitude,
            longitude=longitude,
            time_index=0
        )
    )

    eruption = eruption_service.calculate(
        atmospheric_profile=atmospheric,
        eruption_height=eruption_height,
        eruption_duration=eruption_duration
    )

    particle_summary = (
        particle_service
        .create_particle_summary(
            total_mass=eruption[
                "total_mass"
            ]
        )
    )

    particles = (
        particle_service.create_particles(
            total_mass=eruption[
                "total_mass"
            ],
            latitude=latitude,
            longitude=longitude,
            altitude=altitude
        )
    )

    grid = (
        meteorology_service.nearest_grid(
            latitude=latitude,
            longitude=longitude
        )
    )

    dispersion_result = (
        dispersion.simulate(
            particles=particles,
            duration=duration,
            dt=dt
        )
    )

    trajectory_geojson = (
        geojson_service
        .trajectory_to_feature_collection(
            dispersion_result
        )
    )

    point_geojson = (
        geojson_service
        .trajectory_points_to_feature_collection(
            dispersion_result
        )
    )

    total_particles = sum(
        particle.get(
            "count",
            1
        )
        for particle in particles
    )

    time_information = (
        meteorology_service
        .time_information()
    )

    simulation = {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "duration": duration,
        "dt": dt,
        "grid": grid,
        "meteorology": {
            "time_count": time_information[
                "count"
            ],
            "interval_seconds": time_information[
                "interval_seconds"
            ],
            "interval_hours": time_information[
                "interval_hours"
            ],
            "times": [
                str(time)
                for time in time_information[
                    "times"
                ]
            ]
        },
        "atmospheric": atmospheric,
        "eruption": eruption,
        "particle_summary": particle_summary,
        "particles": {
            "total": total_particles,
            "groups": particles
        },
        "particle_count": (
            dispersion_result[
                "particle_count"
            ]
        ),
        "total_particles": total_particles,
        "steps": dispersion_result[
            "steps"
        ],
        "trajectories": (
            dispersion_result[
                "trajectories"
            ]
        ),
        "geojson": {
            "trajectory": trajectory_geojson,
            "points": point_geojson
        }
    }

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={
            "simulation": simulation
        }
    )
@router.get("/simulations/result", response_class=HTMLResponse)
async def simulation_result(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={}
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
        time_index=0
    )

@router.get("/api/meteorology/stability")
async def meteorology_stability():
    meteorology = MeteorologyService(
        "data/era_merapi_jan_2026.nc"
    )

    atmosphere = meteorology.atmospheric_profile(
        latitude=-7.54,
        longitude=110.44,
        time_index=0
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