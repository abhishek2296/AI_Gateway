from fastapi import APIRouter, Depends

from src.api.dependencies import get_chat_service
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.chat_service import ChatService
from src.schemas.common import APIResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    summary="Chat with the configured LLM",
    description="Send a prompt to the configured language model and receive a response.",
    response_model=APIResponse[ChatResponse],
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):

    result = await chat_service.chat(request.message)

    chat_response = ChatResponse(**result)

    return APIResponse(
        data=chat_response
    )