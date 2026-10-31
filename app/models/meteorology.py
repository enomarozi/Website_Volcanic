from datetime import datetime
from sqlalchemy import DateTime,Float,ForeignKey,Integer,String,Text
from sqlalchemy.orm import Mapped,mapped_column,relationship
from app.database import Base

class MeteorologyDataset(Base):
    __tablename__="meteorology_datasets"

    id:Mapped[int]=mapped_column(Integer,primary_key=True,index=True)
    name:Mapped[str]=mapped_column(String(255),nullable=False)
    description:Mapped[str|None]=mapped_column(Text)
    upper_air_path:Mapped[str|None]=mapped_column(String(1000))
    surface_instant_path:Mapped[str|None]=mapped_column(String(1000))
    surface_accum_path:Mapped[str|None]=mapped_column(String(1000))
    start_time:Mapped[str|None]=mapped_column(String(100))
    end_time:Mapped[str|None]=mapped_column(String(100))
    time_count:Mapped[int|None]=mapped_column(Integer)
    interval_hours:Mapped[float|None]=mapped_column(Float)
    variables:Mapped[str|None]=mapped_column(Text)
    status:Mapped[str]=mapped_column(String(30),default="valid",nullable=False)
    uploaded_by:Mapped[int]=mapped_column(ForeignKey("users.id"),nullable=False)
    created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow,nullable=False)

    uploader=relationship("User",back_populates="meteorology_datasets")
    simulations=relationship("Simulation",back_populates="meteorology_dataset")