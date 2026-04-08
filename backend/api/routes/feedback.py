from fastapi import APIRouter

from api.db import get_db
from api.models.feedback import FeedbackCreate

router = APIRouter()


@router.post("/feedback", status_code=201)
def create_feedback(feedback: FeedbackCreate):
    """Save user feedback to the database."""
    db = get_db()
    feedback_dict = feedback.model_dump()
    result = db.feedback.insert_one(feedback_dict)
    return {"message": "Feedback submitted successfully", "id": str(result.inserted_id)}
