from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    sugerencia: str = Field(..., min_length=1, max_length=1000)


class FeedbackResponse(BaseModel):
    id: int
    sugerencia: str
    created_at: datetime

    model_config = {"from_attributes": True}
