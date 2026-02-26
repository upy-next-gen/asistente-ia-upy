from sqlalchemy.orm import Session

from app.core.logging import logger
from app.db.models.feedback import Feedback


def save_feedback(db: Session, suggestion: str) -> Feedback:
    """
    Persist a user suggestion to the database and return the created record.
    """
    record = Feedback(suggestion=suggestion)
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("Feedback saved | id=%d", record.id)
    return record
