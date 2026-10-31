from fastapi import APIRouter,Depends,Form,Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.services.auth import verify_password

router=APIRouter()
templates=Jinja2Templates(directory="app/templates")

@router.get("/login")
async def login_page(request:Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard",303)
    return templates.TemplateResponse(request=request,name="auth/login.html",context={"request":request})

@router.post("/login")
async def login(request:Request,username:str=Form(...),password:str=Form(...),db:Session=Depends(get_db)):
    user=db.query(User).filter(User.username==username).first()
    if not user or not user.is_active or not verify_password(password,user.password_hash):
        return templates.TemplateResponse(
            request=request,
            name="auth/login.html",
            context={"request":request,"error":"Username atau password salah."},
            status_code=401
        )

    request.session.clear()
    request.session["user_id"]=user.id
    request.session["username"]=user.username
    request.session["role"]=user.role

    return RedirectResponse("/dashboard",303)

@router.get("/logout")
async def logout(request:Request):
    request.session.clear()
    return RedirectResponse("/login",303)