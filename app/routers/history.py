
from fastapi import APIRouter,Depends,Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_user
from app.models.simulation import Simulation

router=APIRouter(prefix="/simulations",tags=["Simulation History"])
templates=Jinja2Templates(directory="app/templates")

@router.get("/history")
def history(request:Request,user=Depends(get_current_user),db:Session=Depends(get_db)):
    if not user: return RedirectResponse("/login",303)
    simulations=db.query(Simulation).filter(Simulation.user_id==user.id).order_by(Simulation.created_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="simulations/history.html",
        context={"request":request,"user":user,"simulations":simulations}
    )
