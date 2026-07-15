from fastapi import HTTPException


class OllamaConnectionException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=503,
            detail="Unable to connect to Ollama. Please ensure Ollama is running."
        )


class LLMResponseException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=500,
            detail="Failed to generate response from the language model."
        )