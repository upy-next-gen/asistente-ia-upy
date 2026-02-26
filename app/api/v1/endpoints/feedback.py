from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.services.feedback_service import save_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a suggestion or feedback",
)
def feedback(request: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackResponse:
    try:
        record = save_feedback(db=db, suggestion=request.sugerencia)
        return FeedbackResponse(
            id=record.id,
            sugerencia=record.suggestion,
            created_at=record.created_at,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save feedback: {exc}",
        ) from exc
