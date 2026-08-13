from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request":request,
        }
    )

@router.get("/simulation/create", response_class=HTMLResponse)
async def create_simulation(request: Request):
    return templates.TemplateResponse(
        "simulation/result.html",
        {
            "request": request,
        }
    )

@router.get("/simulation/history", response_class=HTMLResponse)
async def simulation_history(request: Request):
    return templates.TemplateResponse(
        "simulation/history.html",
        {
            "request": request,
        }
    )

@router.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    return templates.TemplateResponse(
        "reports/index.html",
        {
            "request": request,
        }
    )
