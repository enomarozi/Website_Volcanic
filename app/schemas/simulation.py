
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, model_validator


MAX_ERUPTION_DURATION = 72.0 * 3600.0


class SimulationConfig(BaseModel):
    latitude: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False)
    longitude: float = Field(..., ge=95.0, le=145.0, allow_inf_nan=False)
    altitude: float = Field(0.0, ge=0.0, allow_inf_nan=False)
    eruption_height: float = Field(..., gt=0.0, le=30000.0, allow_inf_nan=False)
    eruption_duration: float = Field(..., gt=0.0, le=MAX_ERUPTION_DURATION, allow_inf_nan=False)
    start_time: datetime | None = None
    time_index: int = Field(0, ge=0)
    vei: int = Field(3, ge=2, le=4)
    duration: float = Field(..., gt=0.0, allow_inf_nan=False)
    total_particles: int = Field(50000, gt=0)
    particle_density: float = Field(2500.0, gt=0.0, allow_inf_nan=False)
    grid_resolution: float = Field(1000.0, gt=0.0, allow_inf_nan=False)
    domain_radius: float = Field(100000.0, gt=0.0, allow_inf_nan=False)
    precipitation: float = Field(0.0, ge=0.0, allow_inf_nan=False)
    mer_method: str = Field("degruyter")

    precipitation_source: Literal["dataset", "manual", "none"] = "dataset"
    precipitation_rate: float | None = Field(None, ge=0.0, allow_inf_nan=False)
    scavenging_a: float = Field(8.4e-5, ge=0.0, allow_inf_nan=False)
    scavenging_b: float = Field(0.79, ge=0.0, allow_inf_nan=False)

    ventilation_rate: float = Field(4000.0, gt=0.0, allow_inf_nan=False)
    mean_pbl_wind: float = Field(7.018, gt=0.0, allow_inf_nan=False)
    convective_velocity: float = Field(2.0, gt=0.0, allow_inf_nan=False)
    timestep_factor: float = Field(1.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_precipitation(self):
        if self.precipitation_source == "manual":
            if self.precipitation_rate is None:
                raise ValueError(
                    "Precipitation rate wajib diisi untuk mode manual."
                )
        return self
