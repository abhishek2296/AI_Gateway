from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.__class__.__name__,
                "message": exc.detail,
            },
        },
    )