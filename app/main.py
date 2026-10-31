from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.database import Base,engine
from app.models import User,MeteorologyDataset,Simulation,SimulationResult
from app.routers.auth import router as auth_router
from app.routers.pages import router as pages_router
from app.routers.meteorology import router as meteorology_router
from app.routers.history import router as history_router

BASE_DIR=Path(__file__).resolve().parents[1]

(BASE_DIR/"data"/"meteorology").mkdir(parents=True,exist_ok=True)
(BASE_DIR/"data"/"simulations").mkdir(parents=True,exist_ok=True)

Base.metadata.create_all(bind=engine)

app=FastAPI(title="Volcanic Ash Dispersion Simulation")

app.add_middleware(
    SessionMiddleware,
    secret_key="CHANGE_THIS_TO_RANDOM_LONG_SECRET"
)

app.include_router(auth_router)
app.include_router(meteorology_router)
app.include_router(history_router)
app.include_router(pages_router)