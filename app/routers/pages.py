from pathlib import Path
import json
import math
from datetime import datetime
from fastapi import APIRouter,Request,Form,Depends,HTTPException
from fastapi.responses import HTMLResponse,RedirectResponse,JSONResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import current_user
from app.models import User,MeteorologyDataset,Simulation,SimulationResult
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
from app.services.noaa_source import acquire_noaa_dataset, SOURCE_CONFIG
import csv
import math
import shutil
import io


router=APIRouter()

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
    
templates=Jinja2Templates(directory="app/templates")
templates.env.filters["sig"] = format_significant
BASE_DIR=Path(__file__).resolve().parents[2]

BACKEND_VEI=3
BACKEND_PARTICLE_DENSITY=2500.0
BACKEND_MER_METHOD="degruyter"



def get_dataset_or_none(db,dataset_id,user):
    dataset=db.get(MeteorologyDataset,dataset_id)
    if not dataset or dataset.status!="valid" or not dataset.upper_air_path:
        return None
    return dataset

def get_simulation_service(dataset:MeteorologyDataset,total_particles=50000,particle_density=BACKEND_PARTICLE_DENSITY,mer_method=BACKEND_MER_METHOD):
    upper_air=BASE_DIR/dataset.upper_air_path
    surface_instant=BASE_DIR/dataset.surface_instant_path if dataset.surface_instant_path else None
    surface_accum=BASE_DIR/dataset.surface_accum_path if dataset.surface_accum_path else None
    meteorology=MeteorologyService(str(upper_air))
    surface_meteorology=None
    if surface_instant and surface_accum:
        surface_meteorology=SurfaceMeteorologyService(str(surface_instant),str(surface_accum))
    turbulence=TurbulenceService()
    eruption=EruptionService(mer_method=mer_method)
    particle=ParticleService(total_particles=total_particles,particle_density=particle_density)
    particle_simulation=ParticleSimulationService(meteorology,turbulence)
    deposition=DepositionService()
    dispersion=DispersionService(particle_simulation,meteorology,deposition,turbulence)
    concentration=ConcentrationService()
    geojson=GeoJSONService()
    service=SimulationService(meteorology,eruption,particle,dispersion,geojson,concentration)
    service.surface_meteorology=surface_meteorology
    return service

def dataset_grid_settings(meteorology,latitude,longitude):
    meteorology.load()
    latitudes=[float(x) for x in meteorology.data["latitude"]]
    longitudes=[float(x) for x in meteorology.data["longitude"]]

    if len(latitudes)<2 or len(longitudes)<2:
        raise ValueError("Dataset meteorologi membutuhkan minimal dua titik latitude dan longitude.")

    lat_step=abs(latitudes[1]-latitudes[0])
    lon_step=abs(longitudes[1]-longitudes[0])
    lat_resolution=lat_step*111320.0
    cos_lat=max(abs(math.cos(math.radians(float(latitude)))),1e-8)
    lon_resolution=lon_step*111320.0*cos_lat
    grid_resolution=(lat_resolution+lon_resolution)/2.0

    source_lat=float(latitude)
    source_lon=float(longitude)
    lat_min=min(latitudes)
    lat_max=max(latitudes)
    lon_min=min(longitudes)
    lon_max=max(longitudes)

    north_distance=abs(source_lat-lat_max)*111320.0
    south_distance=abs(source_lat-lat_min)*111320.0
    east_distance=abs(source_lon-lon_max)*111320.0*cos_lat
    west_distance=abs(source_lon-lon_min)*111320.0*cos_lat

    domain_radius=max(
        min(north_distance,south_distance,east_distance,west_distance),
        grid_resolution
    )

    return {
        "grid_resolution":float(grid_resolution),
        "domain_radius":float(domain_radius),
        "latitude_resolution":float(lat_resolution),
        "longitude_resolution":float(lon_resolution),
        "latitude_min":lat_min,
        "latitude_max":lat_max,
        "longitude_min":lon_min,
        "longitude_max":lon_max
    }

def build_result_display(result,config,dataset,info):
    eruption=result.get("eruption") or {}
    mass=result.get("mass_balance") or {}
    concentration=result.get("concentration") or {}
    deposition=result.get("surface_deposition") or {}
    spatial=result.get("spatial_analysis") or {}
    geojson=result.get("geojson") or {}
    mechanism_mass=mass.get("deposition_mechanism_mass_kg") or {}

    total_mass=float(eruption.get("total_mass") or 0.0)
    deposited=float(mass.get("deposited_mass_kg") or 0.0)
    airborne=float(mass.get("airborne_mass_kg") or 0.0)
    wet=float(mechanism_mass.get("wet") or 0.0)
    dry=float(mechanism_mass.get("dry") or 0.0)
    gravitational=float(mechanism_mass.get("gravitational") or 0.0)

    def percentage(value):
        return round(float(value)/total_mass*100,2) if total_mass>0 else 0.0

    # Hitung column loading dari massa seluruh lapisan vertikal.
    column_values=[]
    time_series=concentration.get("time_series") or []

    if time_series:
        cells=time_series[-1].get("cells") or []
        columns={}

        for cell in cells:
            key=(cell.get("i"),cell.get("j"))
            columns.setdefault(key,{"mass":0.0,"latitude":float(cell["latitude"])})
            columns[key]["mass"]+=float(cell.get("mass") or 0.0)

        earth_radius=6.371e6
        dlat=float(concentration.get("metadata",{}).get("dlat",0.25))
        dlon=float(concentration.get("metadata",{}).get("dlon",0.25))

        for column in columns.values():
            phi=math.radians(column["latitude"])
            area=earth_radius**2*abs(math.cos(phi))*math.radians(dlat)*math.radians(dlon)
            if area>0:
                column_values.append(column["mass"]/area)

    peak_loading=max(column_values) if column_values else None
    mean_loading=sum(column_values)/len(column_values) if column_values else None

    effective_timestep=float((result.get("config") or {}).get("timestep",config.timestep))

    result["location"]={
        "latitude":config.latitude,
        "longitude":config.longitude,
        "altitude":config.altitude
    }

    result["input"]={
        **(result.get("input") or {}),
        "volcano":result.get("volcano_name","Unknown Volcano"),
        "latitude":config.latitude,
        "longitude":config.longitude,
        "altitude":config.altitude,
        "eruption_height":config.eruption_height,
        "eruption_duration":config.eruption_duration,
        "eruption_duration_hours":config.eruption_duration/3600.0,
        "duration":config.duration,
        "duration_hours":config.duration/3600.0,
        "timestep":effective_timestep,
        "output_interval_minutes":effective_timestep/60.0,
        "vei":config.vei,
        "total_particles":config.total_particles,
        "particle_density":config.particle_density,
        "precipitation":config.precipitation,
        "precipitation_mm":config.precipitation*1000.0,
        "grid_resolution":config.grid_resolution,
        "domain_radius":config.domain_radius,
        "mer_method":config.mer_method,
        "time_index":config.time_index,
        "meteorology_dataset":dataset.name
    }

    result["total_erupted_mass"]=total_mass
    result["mer"]=float(eruption.get("mer") or 0.0)
    result["output_interval_minutes"]=round(effective_timestep/60.0,2)

    result["mass_partition"]={
        "airborne":percentage(airborne),
        "wet_deposition":percentage(wet),
        "dry_deposition":percentage(dry),
        "gravitational":percentage(gravitational)
    }

    result["mass_closure"]=mass.get("closure_percent")
    result["peak_column_loading"]=peak_loading
    result["mean_column_loading"]=mean_loading

    result["spatial_analysis"]=spatial
    result["plume_length"]=spatial.get("plume_length_km")
    result["plume_width"]=spatial.get("plume_width_km")
    result["affected_area"]=spatial.get("footprint_km2")
    result["mean_wind_speed"]=eruption.get("mean_wind_speed")

    result["geojson"]=geojson
    result["concentration"]=concentration
    result["surface_deposition"]=deposition

    simulation_id=result.get("simulation_id")
    result["export_pdf_url"]=f"/reports/{simulation_id}/pdf"
    result["export_kmz_url"]=f"/reports/{simulation_id}/kmz"
    result["export_csv_url"]=f"/reports/{simulation_id}/csv"

    return result

@router.get("/",response_class=HTMLResponse)
async def home(request:Request,user:User=Depends(current_user)):
    if not user:
        return RedirectResponse("/login",303)
    return RedirectResponse("/dashboard",303)

@router.get("/dashboard",response_class=HTMLResponse)
async def dashboard(request:Request,user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user:
        return RedirectResponse("/login",303)

    simulations=db.query(Simulation).filter(
        Simulation.user_id==user.id
    ).order_by(Simulation.created_at.desc()).all()

    datasets=db.query(MeteorologyDataset).order_by(
        MeteorologyDataset.created_at.desc()
    ).all()

    total_simulations=len(simulations)
    processing_simulations=sum(1 for s in simulations if s.status=="running")
    completed_simulations=sum(1 for s in simulations if s.status=="completed")
    failed_simulations=sum(1 for s in simulations if s.status=="failed")

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "request":request,
            "user":user,
            "simulations":simulations,
            "datasets":datasets,
            "total_simulations":total_simulations,
            "processing_simulations":processing_simulations,
            "completed_simulations":completed_simulations,
            "failed_simulations":failed_simulations
        }
    )

@router.get("/simulations/{simulation_id}/result",response_class=HTMLResponse)
async def simulation_history_result(
    simulation_id:int,
    request:Request,
    user:User=Depends(current_user),
    db:Session=Depends(get_db)
):
    if not user:
        return RedirectResponse("/login",303)

    simulation=db.get(Simulation,simulation_id)

    if not simulation or simulation.user_id!=user.id:
        return RedirectResponse("/dashboard?error=not_found",303)

    return templates.TemplateResponse(
        request=request,
        name="simulations/history_result.html",
        context={
            "request":request,
            "user":user,
            "simulation":simulation,
            "result":simulation.result
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
            content={"success": False, "message": "Session login berakhir. Silakan login kembali."}
        )

    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == user.id
    ).first()

    if not simulation:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Simulation tidak ditemukan."}
        )

    try:
        db.delete(simulation)
        db.commit()
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "Simulation berhasil dihapus."}
        )
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@router.get("/simulations/create",response_class=HTMLResponse)
async def create_simulation(request:Request,user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user:
        return RedirectResponse("/login",303)

    datasets=db.query(MeteorologyDataset).filter(
        MeteorologyDataset.status=="valid",
        MeteorologyDataset.upper_air_path.is_not(None)
    ).order_by(MeteorologyDataset.created_at.desc()).all()

    return templates.TemplateResponse(
        request=request,
        name="simulations/create.html",
        context={
            "request":request,
            "user":user,
            "datasets":datasets,
            "meteorology":None,
            "error":None
        }
    )

@router.get("/simulations/meteorology/{dataset_id}")
async def simulation_meteorology(dataset_id:int,request:Request,user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user:
        return JSONResponse({"detail":"Unauthorized"},status_code=401)

    dataset=get_dataset_or_none(db,dataset_id,user)
    if not dataset:
        return JSONResponse({"detail":"Meteorology dataset tidak valid."},status_code=404)

    meteorology=None
    try:
        meteorology=MeteorologyService(str(BASE_DIR/dataset.upper_air_path))
        info=meteorology.time_information()
        grid=dataset_grid_settings(meteorology,-7.5407,110.4462)

        surface_info=None
        if dataset.surface_instant_path and dataset.surface_accum_path:
            surface=SurfaceMeteorologyService(
                str(BASE_DIR/dataset.surface_instant_path),
                str(BASE_DIR/dataset.surface_accum_path)
            )
            try:
                surface_data=surface.load()
                surface_info={
                    "time_count":len(surface_data["times"]),
                    "precipitation_units":surface_data["precipitation_units"],
                    "has_precipitation":True,
                    "has_surface_pressure":True,
                    "has_friction_velocity":True,
                    "has_temperature":True,
                    "has_heat_flux":True
                }
            finally:
                surface.close()

        return JSONResponse({
            "dataset_id":dataset.id,
            "name":dataset.name,
            "times":[str(x) for x in info.get("times",[])],
            "time_count":info.get("time_count",info.get("count")),
            "interval_seconds":info.get("interval_seconds"),
            "interval_hours":info.get("interval_hours"),
            "elapsed_seconds":info.get("elapsed_seconds",[]),
            "grid_resolution":grid["grid_resolution"],
            "domain_radius":grid["domain_radius"],
            "latitude_resolution":grid["latitude_resolution"],
            "longitude_resolution":grid["longitude_resolution"],
            "surface":surface_info
        })
    except Exception as exc:
        return JSONResponse({"detail":str(exc)},status_code=400)
    finally:
        if meteorology:
            meteorology.close()

@router.post("/simulations/run",response_class=HTMLResponse)
async def run_simulation(
    request:Request,
    meteorology_dataset_id:int=Form(...),
    name:str=Form("Volcanic Ash Simulation"),
    volcano:str=Form("Merapi"),
    latitude:float=Form(...),
    longitude:float=Form(...),
    altitude:float=Form(...),
    eruption_height:float=Form(...),
    eruption_duration:float=Form(...),
    duration:float=Form(...),
    timestep:float=Form(...),
    time_index:int=Form(0),
    total_particles:int=Form(50000),
    user:User=Depends(current_user),
    db:Session=Depends(get_db)
):
    if not user:
        return RedirectResponse("/login",303)

    dataset=get_dataset_or_none(db,meteorology_dataset_id,user)

    if not dataset:
        datasets=db.query(MeteorologyDataset).filter(
            MeteorologyDataset.status=="valid",
            MeteorologyDataset.upper_air_path.is_not(None)
        ).order_by(MeteorologyDataset.created_at.desc()).all()

        return templates.TemplateResponse(
            request=request,
            name="simulations/create.html",
            context={
                "request":request,
                "user":user,
                "datasets":datasets,
                "meteorology":None,
                "error":"Meteorology dataset tidak valid atau Upper-Air belum tersedia."
            },
            status_code=400
        )

    service=None
    meteorology=None
    surface=None

    try:
        upper_air=BASE_DIR/dataset.upper_air_path
        meteorology=MeteorologyService(str(upper_air))
        info=meteorology.time_information()

        if time_index<0 or time_index>=len(info["times"]):
            raise ValueError(
                f"Time index {time_index} berada di luar range dataset. "
                f"Jumlah waktu tersedia: {len(info['times'])}."
            )

        grid=dataset_grid_settings(meteorology,latitude,longitude)

        if not dataset.surface_instant_path or not dataset.surface_accum_path:
            raise ValueError("Dataset belum memiliki Surface Instant dan Surface Accumulated.")

        surface=SurfaceMeteorologyService(
            str(BASE_DIR/dataset.surface_instant_path),
            str(BASE_DIR/dataset.surface_accum_path)
        )

        surface_profile=surface.surface_profile(latitude,longitude,time_index)
        precipitation=float(surface_profile["precipitation"])

        meteorology.close()
        meteorology=None
        surface.close()
        surface=None

        config=SimulationConfig(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            eruption_height=eruption_height,
            eruption_duration=eruption_duration*3600.0,
            duration=duration*3600.0,
            timestep=timestep,
            time_index=time_index,
            vei=BACKEND_VEI,
            total_particles=total_particles,
            particle_density=BACKEND_PARTICLE_DENSITY,
            grid_resolution=grid["grid_resolution"],
            domain_radius=grid["domain_radius"],
            precipitation=precipitation,
            mer_method=BACKEND_MER_METHOD
        )

        simulation=Simulation(
            user_id=user.id,
            meteorology_dataset_id=dataset.id,
            name=name,
            latitude=config.latitude,
            longitude=config.longitude,
            altitude=config.altitude,
            eruption_height=config.eruption_height,
            eruption_duration=config.eruption_duration,
            duration=config.duration,
            timestep=config.timestep,
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

        service=get_simulation_service(
            dataset=dataset,
            total_particles=config.total_particles,
            particle_density=config.particle_density,
            mer_method=config.mer_method
        )

        info=service.meteorology.time_information()
        result=service.run(config)

        mass=result.get("mass_balance") or {}
        deposition=result.get("surface_deposition") or {}

        validation=MassBalanceService().validate(
            source_mass=mass.get("source_mass_kg",0.0),
            deposited_mass=mass.get("deposited_mass_kg",0.0),
            airborne_mass=mass.get("airborne_mass_kg",0.0),
            grid_deposited_mass=deposition.get("total_deposited_mass",0.0)
        )

        mass["validation"]=validation
        mass["valid"]=validation["valid"]
        mass["grid_valid"]=validation["grid_valid"]
        mass["valid_all"]=validation["valid_all"]
        mass["error"]=validation["error"]
        mass["grid_mass_error"]=validation["grid_mass_error"]

        result["simulation_id"]=simulation.id
        result["volcano_name"]=volcano or "Unknown Volcano"
        result["vei"]=config.vei
        result["simulation_time"]=str(info["times"][config.time_index])

        result["input"]={
            "volcano":volcano or "Unknown Volcano",
            "latitude":config.latitude,
            "longitude":config.longitude,
            "altitude":config.altitude,
            "eruption_height":config.eruption_height,
            "eruption_duration":config.eruption_duration,
            "eruption_duration_hours":config.eruption_duration/3600.0,
            "vei":config.vei,
            "duration":config.duration,
            "duration_hours":config.duration/3600.0,
            "timestep":result["config"]["timestep"],
            "precipitation":config.precipitation,
            "precipitation_m":config.precipitation,
            "precipitation_mm":config.precipitation*1000.0,
            "precipitation_source":f"{dataset.name} — surface accumulated",
            "total_particles":config.total_particles,
            "particle_density":config.particle_density,
            "grid_resolution":config.grid_resolution,
            "domain_radius":config.domain_radius,
            "grid_resolution_source":f"{dataset.name} — meteorological grid",
            "domain_radius_source":f"{dataset.name} — dataset spatial extent",
            "mer_method":config.mer_method,
            "time_index":config.time_index,
            "meteorology_time":str(info["times"][config.time_index]),
            "meteorology_dataset_id":dataset.id,
            "meteorology_dataset":dataset.name
        }
        result=build_result_display(result,config,dataset,info)

        output_dir=BASE_DIR/"data"/"simulations"/str(simulation.id)
        output_dir.mkdir(parents=True,exist_ok=True)

        geojson=result.get("geojson",{})

        files={
            "trajectory_path":("trajectory.geojson",geojson.get("trajectory")),
            "particles_path":("particles.geojson",geojson.get("points")),
            "deposition_path":("deposition.geojson",geojson.get("deposition")),
            "deposition_grid_path":("deposition_grid.geojson",geojson.get("deposition_grid"))
        }

        saved_paths={}

        for key,(filename,data) in files.items():
            if data is None:
                continue
            path=output_dir/filename
            path.write_text(json.dumps(data,default=str),encoding="utf-8")
            saved_paths[key]=str(path.relative_to(BASE_DIR))

        eruption=result.get("eruption") or {}
        mass=result.get("mass_balance") or {}
        deposition=result.get("surface_deposition") or {}

        simulation_result=SimulationResult(
            simulation_id=simulation.id,
            total_mass=eruption.get("total_mass"),
            mer=eruption.get("mer"),
            deposited_mass=mass.get("deposited_mass_kg"),
            airborne_mass=mass.get("airborne_mass_kg"),
            grid_deposited_mass=deposition.get("total_deposited_mass"),
            mass_error=mass.get("error"),
            validation_status=mass.get("valid_all"),
            trajectory_path=saved_paths.get("trajectory_path"),
            particles_path=saved_paths.get("particles_path"),
            deposition_path=saved_paths.get("deposition_path"),
            deposition_grid_path=saved_paths.get("deposition_grid_path")
        )

        db.add(simulation_result)
        simulation.status="completed"
        simulation.completed_at=datetime.utcnow()
        db.commit()

    except Exception as exc:
        if "simulation" in locals() and simulation:
            simulation.status="failed"
            simulation.error_message=str(exc)
            db.commit()
        raise

    finally:
        if service:
            if getattr(service,"meteorology",None):
                service.meteorology.close()
            if getattr(service,"surface_meteorology",None):
                service.surface_meteorology.close()
        if meteorology:
            meteorology.close()
        if surface:
            surface.close()

    return templates.TemplateResponse(
        request=request,
        name="simulations/result.html",
        context={
            "request":request,
            "user":user,
            "result":result,
            "simulation":simulation
        }
    )

@router.get("/meteorology/config",response_class=HTMLResponse)
async def meteorology_config(request:Request,user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user: return RedirectResponse("/login",303)
    datasets=db.query(MeteorologyDataset).filter(MeteorologyDataset.status=="valid").order_by(MeteorologyDataset.created_at.desc()).limit(20).all()
    return templates.TemplateResponse(request=request,name="meteorology/config.html",context={"request":request,"user":user,"datasets":datasets,"sources":SOURCE_CONFIG,"error":None,"message":None})

@router.post("/meteorology/acquire",response_class=HTMLResponse)
async def meteorology_acquire(request:Request,source:str=Form(...),valid_date:str=Form(...),forecast_hour:int=Form(0),latitude:float=Form(...),longitude:float=Form(...),user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user: return RedirectResponse("/login",303)
    try:
        acquired=acquire_noaa_dataset(source,valid_date,forecast_hour,latitude,longitude,user.id)
        from app.services.netcdf_upload import inspect_netcdf
        upper=BASE_DIR/acquired["paths"]["upper"]
        instant=BASE_DIR/acquired["paths"]["instant"]
        accum=BASE_DIR/acquired["paths"]["accum"]
        upper_meta=inspect_netcdf(upper,"upper_air")
        instant_meta=inspect_netcdf(instant,"surface")
        accum_meta=inspect_netcdf(accum,"surface")
        dataset=MeteorologyDataset(name=acquired["name"],description=f"Sumber NOAA {acquired['source']}; cycle {acquired['cycle']}; forecast hour {forecast_hour}.",upper_air_path=acquired["paths"]["upper"],surface_instant_path=acquired["paths"]["instant"],surface_accum_path=acquired["paths"]["accum"],start_time=upper_meta.get("start_time"),end_time=upper_meta.get("end_time"),time_count=upper_meta.get("time_count"),interval_hours=upper_meta.get("interval_hours"),variables=json.dumps(sorted(set(upper_meta.get("variables",[])+instant_meta.get("variables",[])+accum_meta.get("variables",[])))),status="valid",uploaded_by=user.id)
        db.add(dataset); db.commit()
        return RedirectResponse("/meteorology/config?downloaded=1",303)
    except Exception as exc:
        datasets=db.query(MeteorologyDataset).filter(MeteorologyDataset.status=="valid").order_by(MeteorologyDataset.created_at.desc()).limit(20).all()
        return templates.TemplateResponse(request=request,name="meteorology/config.html",context={"request":request,"user":user,"datasets":datasets,"sources":SOURCE_CONFIG,"error":str(exc),"message":None},status_code=400)

@router.get("/source-config",response_class=HTMLResponse)
async def source_config(request:Request,user:User=Depends(current_user)):
    if not user: return RedirectResponse("/login",303)
    return templates.TemplateResponse(request=request,name="simulations/source_config.html",context={"request":request,"user":user})

@router.get("/dispersion",response_class=HTMLResponse)
async def dispersion_config(request:Request,user:User=Depends(current_user)):
    if not user: return RedirectResponse("/login",303)
    return templates.TemplateResponse(request=request,name="simulations/dispersion.html",context={"request":request,"user":user})

@router.get("/reports",response_class=HTMLResponse)
async def reports(request:Request,user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user: return RedirectResponse("/login",303)
    simulations=db.query(Simulation).filter(Simulation.user_id==user.id).order_by(Simulation.created_at.desc()).all()
    return templates.TemplateResponse(request=request,name="reports/index.html",context={"request":request,"user":user,"simulations":simulations})

@router.get("/reports/{simulation_id}/csv")
async def report_csv(simulation_id:int,user:User=Depends(current_user),db:Session=Depends(get_db)):
    if not user: return RedirectResponse("/login",303)
    simulation=db.get(Simulation,simulation_id)
    if not simulation or simulation.user_id!=user.id: return JSONResponse({"detail":"Laporan tidak ditemukan."},status_code=404)
    result=simulation.result
    buffer=io.StringIO(); writer=csv.writer(buffer)
    writer.writerow(["field","value"])
    for key,value in [("simulation_id",simulation.id),("name",simulation.name),("status",simulation.status),("created_at",simulation.created_at),("latitude",simulation.latitude),("longitude",simulation.longitude),("altitude_m",simulation.altitude),("eruption_height_m",simulation.eruption_height),("eruption_duration_s",simulation.eruption_duration),("duration_s",simulation.duration),("timestep_s",simulation.timestep),("precipitation",simulation.precipitation),("particles",simulation.total_particles),("grid_resolution_m",simulation.grid_resolution),("domain_radius_m",simulation.domain_radius)]: writer.writerow([key,value])
    if result:
        for key,value in [("MER_kg_s",result.mer),("total_mass_kg",result.total_mass),("deposited_mass_kg",result.deposited_mass),("airborne_mass_kg",result.airborne_mass),("grid_deposited_mass_kg",result.grid_deposited_mass),("mass_error",result.mass_error),("mass_balance_valid",result.validation_status)]: writer.writerow([key,value])
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]),media_type="text/csv",headers={"Content-Disposition":f"attachment; filename=simulation-{simulation.id}-report.csv"})
