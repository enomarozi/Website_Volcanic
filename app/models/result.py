from datetime import datetime
from sqlalchemy import Boolean,DateTime,Float,ForeignKey,Integer,String
from sqlalchemy.orm import Mapped,mapped_column,relationship
from app.database import Base

class SimulationResult(Base):
    __tablename__="simulation_results"

    id:Mapped[int]=mapped_column(Integer,primary_key=True,index=True)
    simulation_id:Mapped[int]=mapped_column(ForeignKey("simulations.id"),unique=True,nullable=False)

    total_mass:Mapped[float|None]=mapped_column(Float)
    mer:Mapped[float|None]=mapped_column(Float)
    deposited_mass:Mapped[float|None]=mapped_column(Float)
    airborne_mass:Mapped[float|None]=mapped_column(Float)
    grid_deposited_mass:Mapped[float|None]=mapped_column(Float)
    mass_error:Mapped[float|None]=mapped_column(Float)
    validation_status:Mapped[bool|None]=mapped_column(Boolean)

    trajectory_path:Mapped[str|None]=mapped_column(String(1000))
    particles_path:Mapped[str|None]=mapped_column(String(1000))
    deposition_path:Mapped[str|None]=mapped_column(String(1000))
    deposition_grid_path:Mapped[str|None]=mapped_column(String(1000))

    created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow,nullable=False)

    simulation=relationship("Simulation",back_populates="result")