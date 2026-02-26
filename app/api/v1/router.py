from fastapi import APIRouter

from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.feedback import router as feedback_router

api_router = APIRouter()

api_router.include_router(chat_router)
api_router.include_router(feedback_router)
