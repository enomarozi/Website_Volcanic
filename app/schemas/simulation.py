from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):

    volcano_name: str = Field(
        min_length=1
    )

    latitude: float = Field(
        ge=-90,
        le=90
    )

    longitude: float = Field(
        ge=-180,
        le=180
    )

    start_date: str = Field(
        min_length=1
    )

    start_time: str = Field(
        min_length=1
    )

    dataset: str = Field(
        min_length=1
    )

    altitude: float = Field(
        ge=0
    )

    eruption_height: float = Field(
        gt=0
    )

    eruption_duration: float = Field(
        gt=0
    )

    duration: int = Field(
        gt=0
    )

    timestep: int = Field(
        gt=0
    )

    particle_count: int = Field(
        gt=0
    )

    mean_radius: float = Field(
        gt=0
    )

    sigma: float = Field(
        gt=0
    )