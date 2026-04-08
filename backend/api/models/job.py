from pydantic import BaseModel, Field


class JobConfig(BaseModel):
    dataset_name: str
    max_samples: int = Field(default=2000, ge=100, description="Max training samples")
    population_size: int = Field(default=20, ge=5, description="GA population size")
    n_generations: int = Field(default=10, ge=1, description="Number of GA generations")
    bo_calls: int = Field(default=15, ge=10, description="Bayesian optimization calls")
    optimization_mode: str = Field(
        default="multi_3d",
        description="One of: single_f1, multi_2d, multi_3d",
    )
    disable_bo: bool = False
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")