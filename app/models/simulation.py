from datetime import datetime
from sqlalchemy import DateTime,Float,ForeignKey,Integer,String
from sqlalchemy.orm import Mapped,mapped_column,relationship
from app.database import Base

class Simulation(Base):
    __tablename__="simulations"

    id:Mapped[int]=mapped_column(Integer,primary_key=True,index=True)
    user_id:Mapped[int]=mapped_column(ForeignKey("users.id"),nullable=False,index=True)
    meteorology_dataset_id:Mapped[int]=mapped_column(ForeignKey("meteorology_datasets.id"),nullable=False)

    name:Mapped[str]=mapped_column(String(255),nullable=False)
    latitude:Mapped[float]=mapped_column(Float,nullable=False)
    longitude:Mapped[float]=mapped_column(Float,nullable=False)
    altitude:Mapped[float]=mapped_column(Float,nullable=False)
    eruption_height:Mapped[float]=mapped_column(Float,nullable=False)
    eruption_duration:Mapped[float]=mapped_column(Float,nullable=False)
    duration:Mapped[float]=mapped_column(Float,nullable=False)
    timestep:Mapped[float]=mapped_column(Float,nullable=False)
    time_index:Mapped[int]=mapped_column(Integer,nullable=False)
    precipitation:Mapped[float]=mapped_column(Float,default=0.0,nullable=False)
    total_particles:Mapped[int]=mapped_column(Integer,nullable=False)
    particle_density:Mapped[float]=mapped_column(Float,nullable=False)
    grid_resolution:Mapped[float]=mapped_column(Float,nullable=False)
    domain_radius:Mapped[float]=mapped_column(Float,nullable=False)
    mer_method:Mapped[str]=mapped_column(String(50),nullable=False)
    vei:Mapped[int]=mapped_column(Integer,nullable=False)

    status:Mapped[str]=mapped_column(String(30),default="pending",nullable=False,index=True)
    error_message:Mapped[str|None]=mapped_column(String(2000))
    created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow,nullable=False)
    completed_at:Mapped[datetime|None]=mapped_column(DateTime)

    user=relationship("User",back_populates="simulations")
    meteorology_dataset=relationship("MeteorologyDataset",back_populates="simulations")
    result=relationship("SimulationResult",back_populates="simulation",uselist=False,cascade="all,delete-orphan")