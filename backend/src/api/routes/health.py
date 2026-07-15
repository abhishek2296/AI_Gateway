from fastapi import APIRouter, Depends

from src.api.dependencies import get_llm_service
from src.schemas.common import APIResponse
from src.schemas.health import HealthResponse
from src.services.base_llm import BaseLLMService

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    summary="Check AI Gateway health",
    description="Checks connectivity with the configured LLM provider.",
    response_model=APIResponse[HealthResponse],
)
async def health(
    llm: BaseLLMService = Depends(get_llm_service),
):

    result = await llm.check_connection()

    return APIResponse(
        data=HealthResponse(**result)
    )