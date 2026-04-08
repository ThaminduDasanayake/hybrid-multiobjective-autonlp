from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    name: str | None = None
    message: str
