from fastapi import Depends,Request
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User

def get_current_user(request:Request,db:Session=Depends(get_db)):
    user_id=request.session.get("user_id")
    if not user_id:
        return None
    user=db.get(User,int(user_id))
    if not user or not user.is_active:
        request.session.clear()
        return None
    return user

current_user=get_current_user