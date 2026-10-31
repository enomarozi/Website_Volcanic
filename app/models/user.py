from datetime import datetime
from sqlalchemy import Boolean,DateTime,Integer,String
from sqlalchemy.orm import Mapped,mapped_column,relationship
from app.database import Base

class User(Base):
    __tablename__="users"

    id:Mapped[int]=mapped_column(Integer,primary_key=True,index=True)
    username:Mapped[str]=mapped_column(String(100),unique=True,index=True,nullable=False)
    email:Mapped[str|None]=mapped_column(String(255),unique=True,index=True)
    password_hash:Mapped[str]=mapped_column(String(255),nullable=False)
    role:Mapped[str]=mapped_column(String(30),default="user",nullable=False)
    is_active:Mapped[bool]=mapped_column(Boolean,default=True,nullable=False)
    created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow,nullable=False)

    simulations=relationship("Simulation",back_populates="user",cascade="all,delete-orphan")
    meteorology_datasets=relationship("MeteorologyDataset",back_populates="uploader",cascade="all,delete-orphan")