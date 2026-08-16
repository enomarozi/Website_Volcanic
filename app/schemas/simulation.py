from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    volcano_name: str = Field(min_length=1)
    latitude: float
    longitude: float
    start_date: str
    start_time: str
    duration: int = Field(gt=0)
    dataset: str
    particle_count: int = Field(gt=0)
    mean_radius: float = Field(gt=0)
    sigma: float = Field(gt=0)
    timestep: int = Field(gt=0)