# OpenAI Provider

Registry key: `openai`

## Configuration

| Setting / env | Purpose |
|---------------|---------|
| `OPENAI_API_KEY` | API key (via `ProviderConfigResolver` → `api_key_env` on DB row) |
| `ProviderConfiguration.endpoint` | Optional custom base URL (default `https://api.openai.com/v1`) |
| `PROVIDER_HTTP_MAX_RETRIES` | Retry count for 429/502/503/504 |
| `PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS` | Model catalog cache TTL |

Constructor kwargs from `ProviderConfigResolver`: `api_key`, `base_url`, `timeout`, optional `organization`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chat/completions` | Chat (sync + SSE stream) |
| POST | `/embeddings` | Embeddings |
| GET | `/models` | Model catalog + health probe |

## Capabilities

| Capability | Supported |
|------------|-----------|
| Chat | Yes |
| Streaming | Yes (SSE) |
| Embeddings | Yes |
| Vision | Yes (`ImagePart` → `image_url`) |
| Tools | Yes |
| Structured JSON | Yes (`response_format`) |
| `validate_model` | Yes (cached catalog) |
| `estimate_tokens` | No (`UnsupportedCapabilityError`) |

## Auth

`Authorization: Bearer {api_key}`; optional `OpenAI-Organization` header.
