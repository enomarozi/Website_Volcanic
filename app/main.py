from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers.pages import router as pages_router

app = FastAPI(
    title="Volcanic Ash Analysis",
    description="Web-based volcanic ash dispersion analysis",
    version="0.1.0"
)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)

app.include_router(pages_router)

