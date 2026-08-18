from fastapi import APIRouter, HTTPException, Query

from app.services.meteorology import MeteorologyService


router = APIRouter(
    prefix="/api/meteorology",
    tags=["Meteorology"]
)


# =========================================================
# SERVICE
# =========================================================

meteorology_service = None


def configure_meteorology_service(
    service: MeteorologyService
):
    global meteorology_service

    meteorology_service = service


def get_service() -> MeteorologyService:

    if meteorology_service is None:

        raise HTTPException(
            status_code=500,
            detail="Meteorology service is not configured."
        )

    return meteorology_service


# =========================================================
# DATASET INFORMATION
# =========================================================

@router.get("/info")
async def meteorology_info():

    service = get_service()

    return service.dataset_information()


# =========================================================
# TIME INFORMATION
# =========================================================

@router.get("/times")
async def meteorology_times():

    service = get_service()

    information = (
        service.time_information()
    )

    return {
        "count": information["count"],
        "interval_seconds": information[
            "interval_seconds"
        ],
        "interval_hours": information[
            "interval_hours"
        ],
        "duration_seconds": information[
            "duration_seconds"
        ],
        "duration_hours": information[
            "duration_hours"
        ],
        "times": [
            str(time)
            for time in information["times"]
        ],
        "elapsed_seconds": information[
            "elapsed_seconds"
        ]
    }


# =========================================================
# PRESSURE LEVELS
# =========================================================

@router.get("/levels")
async def meteorology_levels():

    service = get_service()

    return {
        "count": len(
            service.pressure_level
        ),
        "levels": [
            float(level)
            for level in service.pressure_level
        ]
    }


# =========================================================
# GRID
# =========================================================

@router.get("/grid")
async def meteorology_grid():

    service = get_service()

    return {
        "latitude": [
            float(value)
            for value in service.latitude
        ],
        "longitude": [
            float(value)
            for value in service.longitude
        ],
        "latitude_count": len(
            service.latitude
        ),
        "longitude_count": len(
            service.longitude
        )
    }


# =========================================================
# NEAREST GRID
# =========================================================

@router.get("/nearest-grid")
async def nearest_grid(
    latitude: float = Query(...),
    longitude: float = Query(...)
):

    service = get_service()

    return service.nearest_grid(
        latitude=latitude,
        longitude=longitude
    )


# =========================================================
# ATMOSPHERIC PROFILE
# =========================================================

@router.get("/profile")
async def atmospheric_profile(
    latitude: float = Query(...),
    longitude: float = Query(...),
    time_index: int = Query(
        default=0,
        ge=0
    )
):

    service = get_service()

    try:

        return service.atmospheric_profile(
            latitude=latitude,
            longitude=longitude,
            time_index=time_index
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# =========================================================
# WIND AT ALTITUDE
# =========================================================

@router.get("/wind")
async def wind_at_altitude(
    latitude: float = Query(...),
    longitude: float = Query(...),
    altitude: float = Query(...),
    time_index: int = Query(
        default=0,
        ge=0
    )
):

    service = get_service()

    try:

        return service.wind_at_altitude(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# =========================================================
# METEOROLOGICAL STATE
# =========================================================

@router.get("/state")
async def meteorological_state(
    latitude: float = Query(...),
    longitude: float = Query(...),
    altitude: float = Query(...),
    time_index: int = Query(
        default=0,
        ge=0
    )
):

    service = get_service()

    try:

        return service.meteorological_state(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            time_index=time_index
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )