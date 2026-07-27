# Anthropic Provider

Registry key: `anthropic`

## Configuration

| Setting / env | Purpose |
|---------------|---------|
| `ANTHROPIC_API_KEY` | API key (via DB `api_key_env`) |
| `ANTHROPIC_API_VERSION` | Messages API version header (default `2023-06-01`) |
| `ProviderConfiguration.endpoint` | Optional custom base URL (default `https://api.anthropic.com`) |

Constructor kwargs: `api_key`, `base_url`, `timeout`, `api_version`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/messages` | Chat (sync + SSE stream) |
| GET | `/v1/models` | Model catalog + health probe |

## Capabilities

| Capability | Supported |
|------------|-----------|
| Chat | Yes |
| Streaming | Yes (SSE event types) |
| Embeddings | No (`UnsupportedCapabilityError`) |
| Vision | Yes (base64 `ImagePart`) |
| Tools | Yes (`tool_use` / `tool_result` blocks) |
| Extended thinking | Metadata in `ChatResponse.details["thinking"]` |
| Structured JSON | Yes (`response_format.type=json`) |
| `validate_model` | Yes |

## Auth

`x-api-key: {api_key}` and `anthropic-version: {ANTHROPIC_API_VERSION}`.
