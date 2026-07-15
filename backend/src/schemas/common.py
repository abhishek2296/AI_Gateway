from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorResponse(BaseModel):
    code: str
    message: str


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T


class FailureResponse(BaseModel):
    success: bool = False
    error: ErrorResponse

class HealthResponse(BaseModel):
    status:str