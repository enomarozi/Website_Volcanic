
from datetime import datetime
from pydantic import BaseModel, Field


MAX_ERUPTION_DURATION = 72.0 * 3600.0


class SimulationConfig(BaseModel):
    latitude: float = Field(..., ge=-10.0, le=10.0)
    longitude: float = Field(..., ge=95.0, le=145.0)
    altitude: float = Field(0.0, ge=0.0)
    eruption_height: float = Field(..., gt=0.0, le=30000.0)
    eruption_duration: float = Field(..., gt=0.0, le=MAX_ERUPTION_DURATION)
    start_time: datetime | None = None
    time_index: int = Field(0, ge=0)
    vei: int = Field(3, ge=2, le=4)
    duration: float = Field(..., gt=0.0)
    total_particles: int = Field(50000, gt=0)
    particle_density: float = Field(2500.0, gt=0.0)
    grid_resolution: float = Field(1000.0, gt=0.0)
    domain_radius: float = Field(100000.0, gt=0.0)
    precipitation: float = Field(0.0, ge=0.0)
    mer_method: str = Field("degruyter")
