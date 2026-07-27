# Gemini Provider

Registry key: `gemini`

## Configuration

| Setting / env | Purpose |
|---------------|---------|
| `GEMINI_API_KEY` | API key (via DB `api_key_env`) |
| `ProviderConfiguration.endpoint` | Optional custom base URL (default `https://generativelanguage.googleapis.com`) |

Constructor kwargs: `api_key`, `base_url`, `timeout`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1beta/models/{model}:generateContent` | Chat |
| POST | `/v1beta/models/{model}:streamGenerateContent` | Streaming (`alt=sse`) |
| POST | `/v1beta/models/{model}:embedContent` | Embeddings |
| GET | `/v1beta/models` | Model catalog + health probe |

Model IDs are normalized to `models/{name}` when the `models/` prefix is omitted.

## Capabilities

| Capability | Supported |
|------------|-----------|
| Chat | Yes |
| Streaming | Yes (SSE or NDJSON) |
| Embeddings | Yes (`embedContent`) |
| Vision | Yes (`inlineData` from base64 `ImagePart`) |
| Tools | Yes (`functionDeclarations` / `functionCall`) |
| Structured JSON | Yes (`generationConfig.responseSchema`) |
| Safety settings | Yes via `ChatRequest.provider_options["safetySettings"]` |
| `validate_model` | Yes |

## Auth

`x-goog-api-key: {api_key}` header on all requests.
