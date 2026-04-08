from pydantic import BaseModel, Field


class AblationConfig(BaseModel):
    mode: str = Field(
        default="multi_3d",
        description="Optimization mode: single_f1 | multi_2d | multi_3d | random_search",
    )
    disable_bo: bool = Field(default=False, description="Disable Bayesian optimization")
    parent_job_id: str = Field(..., description="Job ID whose config to inherit")
