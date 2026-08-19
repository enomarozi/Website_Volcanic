from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    volcano_name: str
    latitude: float
    longitude: float
    start_date: str
    start_time: str
    dataset: str
    time_index: int = Field(ge=0)
    altitude: float = Field(ge=0)
    eruption_height: float = Field(ge=0)
    eruption_duration: float = Field(gt=0)
    particle_count: int = Field(gt=0)
    mean_radius: float = Field(gt=0)
    sigma: float = Field(gt=0)
    duration: int = Field(gt=0)
    timestep: int = Field(gt=0)