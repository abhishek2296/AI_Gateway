"""
Abstract provider contract for multi-vendor LLM backends.

This module is the heart of the provider abstraction layer described in
``05-ai-gateway-architecture.mdc``: it defines the vendor-neutral data
transfer objects (``ChatRequest``, ``ChatResponse``, ``ModelInfo``, etc.)
that flow between the gateway's service layer and every provider adapter,
plus the abstract ``BaseProvider`` contract every concrete adapter
(``OllamaProvider``, ``OpenAIProvider``, ``AnthropicProvider``,
``GeminiProvider``) must implement.

Nothing in this module talks to a specific vendor's SDK or wire format —
that translation work happens entirely inside each concrete adapter
(``ollama.py``, ``openai.py``, etc.), which map these normalized types
to/from vendor-specific JSON payloads. Keeping that mapping logic out of
this module is what lets the service/route layers depend only on these
types and never on vendor-specific shapes, satisfying the "routes never
import provider SDKs directly" rule from the architecture doc.

All DTOs here are immutable (``frozen=True``) and memory-lean (``slots=True``)
dataclasses: normalized request/response objects are created once, passed
around, and never mutated in place, so immutability catches accidental
mutation bugs early and ``slots=True`` avoids the per-instance ``__dict__``
overhead for what can be a high-volume type (e.g. one ``ChatStreamChunk`` per
streamed token).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TextPart:
    """
    Text content block within a multimodal message.

    One of the two possible members of :data:`ContentPart`, used when a
    :class:`ChatMessage` needs to mix plain text with other content types
    (e.g. text followed by an image) in a single message via
    ``ChatMessage.parts``. For messages that are text-only, using the
    simpler ``ChatMessage.content`` string field is preferred over wrapping
    it in a single ``TextPart``.

    Attributes:
        text: The literal text content of this block.

    Example:
        >>> TextPart(text="Describe this image:")
        TextPart(text='Describe this image:')
    """

    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """
    Image content block within a multimodal message.

    The other member of :data:`ContentPart`. Supports two mutually-exclusive
    ways of referencing image data — a remote ``url`` or inline
    ``base64_data`` — since different vendors and use cases favor one or the
    other (e.g. a publicly hosted image vs. a locally uploaded file that
    hasn't been persisted anywhere). Concrete adapters decide how to map
    whichever field is populated into their vendor's expected image
    format (e.g. OpenAI's ``image_url`` content block vs. Anthropic's
    base64 ``source`` block).

    Attributes:
        media_type: The MIME type of the image, e.g. ``"image/jpeg"`` or
            ``"image/png"``. Defaults to ``"image/jpeg"`` since that's the
            most common format for photos; override explicitly for other
            formats (particularly required for ``base64_data`` images, since
            without a URL there is no other way to infer the format).
        url: A publicly reachable URL pointing at the image, or ``None`` if
            the image is instead supplied inline via ``base64_data``.
        base64_data: The base64-encoded image bytes (no ``data:`` URL prefix
            — adapters add that themselves if the vendor's API expects a
            data URL), or ``None`` if ``url`` is used instead.

    Example:
        >>> ImagePart(url="https://example.com/cat.jpg")
        ImagePart(media_type='image/jpeg', url='https://example.com/cat.jpg', base64_data=None)
        >>> ImagePart(media_type="image/png", base64_data="aGVsbG8=")
        ImagePart(media_type='image/png', url=None, base64_data='aGVsbG8=')
    """

    media_type: str = "image/jpeg"
    url: str | None = None
    base64_data: str | None = None


ContentPart = TextPart | ImagePart
"""Union of every supported multimodal content block type for a :class:`ChatMessage`."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """
    Function tool schema exposed to the model.

    Describes one callable "tool"/"function" that the model may choose to
    invoke during a chat completion (function calling). Passed via
    ``ChatRequest.tools``; each concrete adapter maps this to its vendor's
    tool-schema format (e.g. OpenAI's ``{"type": "function", "function":
    {...}}`` wrapper vs. Anthropic's flatter ``{"name", "input_schema"}``
    shape vs. Gemini's ``functionDeclarations`` list).

    Attributes:
        name: The tool's callable name, as the model will reference it when
            requesting an invocation (surfaces later as ``ToolCall.name``).
        description: A natural-language explanation of what the tool does
            and when to use it, shown to the model to help it decide whether
            /how to call it. ``None`` if no description is provided.
        parameters: A JSON Schema object (as a plain mapping) describing the
            tool's expected arguments, or ``None`` if the tool takes no
            arguments / the schema isn't specified.

    Example:
        >>> ToolDefinition(
        ...     name="get_weather",
        ...     description="Get the current weather for a city.",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"city": {"type": "string"}},
        ...         "required": ["city"],
        ...     },
        ... )
        ToolDefinition(name='get_weather', description='Get the current weather for a city.', parameters={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']})
    """

    name: str
    description: str | None = None
    parameters: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    A tool invocation requested by the model.

    Appears on ``ChatResponse.tool_calls``/``ChatStreamChunk.tool_calls``
    when the model decides to call one or more tools instead of (or in
    addition to) returning plain text, and again on an outgoing
    ``ChatMessage.tool_calls`` when replaying the assistant's prior tool-call
    turn back to the model as conversation history.

    Attributes:
        id: A vendor-assigned identifier for this specific tool call,
            correlating it with the ``ToolResult`` sent back in a later
            turn (via ``ToolResult.tool_call_id`` /
            ``ChatMessage.tool_call_id``). Some vendors (e.g. Gemini) don't
            provide a distinct call id, in which case the adapter maps the
            tool name into this field instead.
        name: The name of the tool being invoked, matching a
            ``ToolDefinition.name`` from the original request.
        arguments: The tool call arguments, as a raw JSON-encoded string
            (not a parsed dict) — kept as a string here because that's the
            wire format most vendors (OpenAI, Anthropic) use for
            streaming-friendly incremental argument accumulation; callers
            that need a parsed dict decode this themselves.

    Example:
        >>> ToolCall(id="call_1", name="get_weather", arguments='{"city": "Paris"}')
        ToolCall(id='call_1', name='get_weather', arguments='{"city": "Paris"}')
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    Result returned to the model after tool execution.

    Represents the gateway/client executing a tool the model requested (via
    a ``ToolCall``) and sending the result back so the model can continue
    the conversation with that information.

    Attributes:
        tool_call_id: Must match the ``ToolCall.id`` this is a response to,
            so the model (and vendor API) can correlate the result with the
            specific call that produced it — required when a single
            assistant turn requested multiple tool calls at once.
        content: The tool's output, as a string (e.g. a JSON-encoded result
            or plain text), to be shown to the model.

    Example:
        >>> ToolResult(tool_call_id="call_1", content='{"temperature_c": 18}')
        ToolResult(tool_call_id='call_1', content='{"temperature_c": 18}')
    """

    tool_call_id: str
    content: str


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """
    Structured output configuration for chat completions.

    Lets a caller request the model constrain its output to plain text,
    generic JSON, or JSON conforming to a specific schema. Mapped by each
    adapter into its vendor's structured-output mechanism (e.g. OpenAI's
    ``response_format``, Gemini's ``responseMimeType``/``responseSchema``);
    Anthropic only supports the plain-JSON case in this codebase's mapping
    (see ``anthropic._build_messages_payload``), since the Messages API's
    schema-constrained output mechanism differs enough that it isn't
    currently wired up.

    Attributes:
        type: Which structured-output mode to request:
            ``"text"`` for unconstrained plain text (the default),
            ``"json"`` for "valid JSON, but no specific schema enforced", or
            ``"json_schema"`` for "JSON conforming to ``json_schema``".
        json_schema: The JSON Schema (as a plain mapping) the response must
            conform to, required when ``type == "json_schema"`` and ignored
            otherwise. ``None`` when not applicable.

    Example:
        >>> ResponseFormat(type="json")
        ResponseFormat(type='json', json_schema=None)
        >>> ResponseFormat(
        ...     type="json_schema",
        ...     json_schema={"schema": {"type": "object", "properties": {"answer": {"type": "string"}}}},
        ... )
        ResponseFormat(type='json_schema', json_schema={'schema': {'type': 'object', 'properties': {'answer': {'type': 'string'}}}})
    """

    type: Literal["text", "json", "json_schema"] = "text"
    json_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """
    One message in a chat completion request.

    A single turn in the conversation history sent as part of a
    :class:`ChatRequest`. Supports four distinct "shapes" a message can take,
    only one of which is normally populated per instance:

    1. A plain text turn — only ``role``/``content`` set.
    2. A multimodal turn — ``parts`` set (text + images), with ``content``
       optionally holding a leading text portion (adapter-dependent).
    3. An assistant turn that called tools — ``tool_calls`` set.
    4. A tool-result turn responding to a prior tool call — ``tool_call_id``
       set (with ``content`` holding the tool's output).

    Which combination of fields is populated is what each adapter's
    ``_map_message`` function inspects to decide how to serialize the
    message for its vendor's API.

    Attributes:
        role: The message author's role, e.g. ``"system"``, ``"user"``,
            ``"assistant"``, or ``"tool"``. Not constrained to a
            ``Literal`` here because different vendors recognize slightly
            different role vocabularies (e.g. Gemini has no ``"system"``
            role and instead extracts it into a separate top-level field —
            see ``gemini._build_generate_payload``).
        content: The plain-text content of the message. Defaults to
            ``""`` since some message shapes (e.g. a pure tool-call turn)
            carry their real payload in ``tool_calls`` instead.
        parts: An ordered sequence of multimodal :data:`ContentPart` blocks
            (text and/or images), or ``None`` for a plain-text-only message.
        tool_calls: The tool calls this (assistant) message is making, or
            ``None`` if this message isn't a tool-call turn.
        tool_call_id: The id of the ``ToolCall`` this message is a result
            for, or ``None`` if this message isn't a tool-result turn.

    Example:
        >>> ChatMessage(role="user", content="Hello!")
        ChatMessage(role='user', content='Hello!', parts=None, tool_calls=None, tool_call_id=None)
        >>> ChatMessage(
        ...     role="user",
        ...     parts=(TextPart(text="What's in this image?"), ImagePart(url="https://example.com/x.jpg")),
        ... )
        ChatMessage(role='user', content='', parts=(TextPart(text="What's in this image?"), ImagePart(media_type='image/jpeg', url='https://example.com/x.jpg', base64_data=None)), tool_calls=None, tool_call_id=None)
    """

    role: str
    content: str = ""
    parts: Sequence[ContentPart] | None = None
    tool_calls: Sequence[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """
    Token counts returned by a provider completion.

    Normalizes each vendor's own usage-accounting field names (OpenAI's
    ``prompt_tokens``/``completion_tokens``, Anthropic's
    ``input_tokens``/``output_tokens``, Gemini's
    ``promptTokenCount``/``candidatesTokenCount``, Ollama's
    ``prompt_eval_count``/``eval_count``) into one common shape.

    Attributes:
        prompt_tokens: Number of tokens consumed by the input/prompt, or
            ``None`` if the vendor didn't report it for this call.
        completion_tokens: Number of tokens generated in the response, or
            ``None`` if not reported.
        total_tokens: Combined prompt + completion token count. Some
            vendors report this directly; others (Ollama, Anthropic) don't,
            in which case the adapter computes it itself as the sum of the
            two fields above (when both are available).

    Example:
        >>> TokenUsage(prompt_tokens=12, completion_tokens=34, total_tokens=46)
        TokenUsage(prompt_tokens=12, completion_tokens=34, total_tokens=46)
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """
    Normalized chat completion input.

    Provider adapters translate this contract into vendor-specific API payloads.

    This is the single input shape the gateway's service layer builds and
    passes to *any* provider's ``chat``/``stream_chat`` method — the service
    layer never needs to know which vendor it's talking to when constructing
    a request. Fields that a given vendor doesn't support (e.g. Anthropic
    has no ``embeddings`` API entirely; some vendors ignore ``tool_choice``)
    are simply not mapped by that vendor's ``_build_*_payload`` function
    rather than raising — since silently ignoring an unsupported *request
    tuning knob* (as opposed to an unsupported *capability*, like calling
    ``embeddings`` on Anthropic) is generally more useful than failing outright.

    Attributes:
        model: The vendor-specific model identifier to use, e.g.
            ``"gpt-4o"``, ``"claude-3-5-sonnet-20241022"``, ``"qwen3:8b"``.
        messages: The ordered conversation history to send, oldest first.
        temperature: Sampling temperature (higher = more random), in the
            vendor's expected range (commonly 0.0-2.0). ``None`` lets the
            vendor use its own default.
        max_tokens: Maximum number of tokens to generate in the response.
            ``None`` lets the vendor use its own default/limit. Note
            Anthropic's API requires this field (unlike OpenAI/Gemini,
            where it's optional), so ``AnthropicProvider`` substitutes a
            fallback value of ``1024`` when ``None`` (see
            ``anthropic._build_messages_payload``).
        stream: Whether this request is being used for a streaming call.
            ``ChatRequest`` doesn't dispatch between ``chat``/``stream_chat``
            itself — the caller picks which method to call — this flag
            exists so the same request object's data can be reused to build
            either a streaming or non-streaming vendor payload consistently.
        top_p: Nucleus sampling threshold, typically in ``[0, 1]``. ``None``
            lets the vendor use its own default.
        stop: Sequences that should cause the model to stop generating
            further tokens, or ``None`` for no custom stop sequences.
        tools: The tool definitions available for the model to call, or
            ``None``/empty if function calling isn't being used for this
            request.
        tool_choice: Vendor-specific hint controlling whether/which tool the
            model must call (e.g. ``"auto"``, ``"none"``, a specific tool
            name), or ``None`` to let the vendor default (usually
            ``"auto"``).
        response_format: Structured output configuration, or ``None`` for
            unconstrained plain-text output.
        provider_options: An escape hatch for vendor-specific parameters
            that don't have a normalized field here (e.g. Gemini's
            ``safetySettings`` — see ``gemini._build_generate_payload``).
            ``None`` if no vendor-specific options are needed. Deliberately
            typed as a raw ``Mapping`` rather than individually modeled,
            since exhaustively modeling every vendor-specific knob here
            would defeat the purpose of a normalized contract.

    Example:
        >>> request = ChatRequest(
        ...     model="gpt-4o",
        ...     messages=[ChatMessage(role="user", content="Hi there")],
        ...     temperature=0.7,
        ...     max_tokens=256,
        ... )
        >>> request.model
        'gpt-4o'
        >>> request.stream
        False
    """

    model: str
    messages: Sequence[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None
    stop: Sequence[str] | None = None
    tools: Sequence[ToolDefinition] | None = None
    tool_choice: str | None = None
    response_format: ResponseFormat | None = None
    provider_options: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """
    Normalized non-streaming chat completion output.

    Returned by ``BaseProvider.chat`` after each concrete adapter maps its
    vendor's full JSON response body into this common shape.

    Attributes:
        content: The complete generated text response. Empty string if the
            model only returned tool calls with no accompanying text.
        model: The model identifier the vendor reports actually served the
            request (may differ slightly from the requested ``model``, e.g.
            a vendor resolving an alias to a specific dated snapshot).
        finish_reason: Vendor-reported reason generation stopped (e.g.
            ``"stop"``, ``"length"``, ``"tool_calls"``), or ``None`` if not
            reported.
        usage: Token accounting for this completion, or ``None`` if the
            vendor didn't report any usage data.
        provider_response_id: The vendor's own identifier for this specific
            response/generation (useful for support requests / vendor-side
            debugging), or ``None`` if not provided (Ollama doesn't return
            one).
        tool_calls: The tool calls the model requested, or ``None`` if the
            model didn't call any tools.
        details: A free-form string-to-string bag for vendor-specific extras
            that don't warrant a normalized field, e.g. Anthropic's
            "thinking" (extended reasoning) output text. Defaults to an
            empty dict rather than ``None`` so callers can always safely do
            ``response.details.get(...)`` without a ``None`` check.

    Example:
        >>> ChatResponse(content="Hello!", model="gpt-4o", finish_reason="stop")
        ChatResponse(content='Hello!', model='gpt-4o', finish_reason='stop', usage=None, provider_response_id=None, tool_calls=None, details={})
    """

    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_response_id: str | None = None
    tool_calls: Sequence[ToolCall] | None = None
    details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    """
    One incremental chunk from a streaming chat completion.

    Yielded repeatedly by ``BaseProvider.stream_chat`` as a response is
    generated. Unlike :class:`ChatResponse`, most fields on a given chunk are
    typically empty/``None`` — a chunk carrying new text usually only
    populates ``content``, while the final chunk of a stream typically
    populates ``finish_reason``/``usage`` instead (mirroring how vendors
    themselves split "content deltas" from "final metadata" across separate
    SSE events — see ``streaming.py`` and each adapter's ``_map_stream_chunk``
    /``_map_stream_event``).

    Attributes:
        content: The incremental text delta carried by this chunk. Empty
            string (not ``None``) when this chunk carries no new text (e.g.
            a chunk that only reports the final ``finish_reason``), so
            callers can always safely concatenate ``content`` across chunks
            without a ``None`` check.
        finish_reason: Populated only on the terminal chunk(s) of the
            stream, once the vendor reports generation is complete;
            ``None`` for all preceding chunks.
        usage: Token accounting, populated only once the vendor reports it
            (typically alongside ``finish_reason`` on/near the final
            chunk(s)); ``None`` otherwise.
        tool_calls: Tool call information carried by this chunk. Depending
            on the vendor, tool call arguments may arrive incrementally
            across multiple chunks (accumulated by the caller) rather than
            complete in one chunk.

    Example:
        >>> ChatStreamChunk(content="Hel")
        ChatStreamChunk(content='Hel', finish_reason=None, usage=None, tool_calls=None)
        >>> ChatStreamChunk(content="", finish_reason="stop", usage=TokenUsage(total_tokens=42))
        ChatStreamChunk(content='', finish_reason='stop', usage=TokenUsage(prompt_tokens=None, completion_tokens=None, total_tokens=42), tool_calls=None)
    """

    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    tool_calls: Sequence[ToolCall] | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingsRequest:
    """
    Normalized embeddings input.

    Attributes:
        model: The vendor-specific embeddings model identifier, e.g.
            ``"text-embedding-3-small"``.
        input: The text to embed — either a single string or a sequence of
            strings for batch embedding in one call. Adapters that only
            support one input at a time (e.g. ``GeminiProvider.embeddings``,
            which calls a single-content ``embedContent`` endpoint) wrap a
            lone string into a one-element list internally as needed.

    Example:
        >>> EmbeddingsRequest(model="text-embedding-3-small", input="Hello world")
        EmbeddingsRequest(model='text-embedding-3-small', input='Hello world')
        >>> EmbeddingsRequest(model="text-embedding-3-small", input=["a", "b"])
        EmbeddingsRequest(model='text-embedding-3-small', input=['a', 'b'])
    """

    model: str
    input: str | Sequence[str]


@dataclass(frozen=True, slots=True)
class EmbeddingsResponse:
    """
    Normalized embeddings output.

    Attributes:
        model: The model identifier that generated the embeddings.
        embeddings: One vector per input, in the same order as the request's
            ``input`` (or a single-element list, for adapters that only
            embed one input at a time).
        usage: Token accounting for the embedding call, or ``None`` if not
            reported by the vendor.

    Example:
        >>> EmbeddingsResponse(model="text-embedding-3-small", embeddings=[[0.1, 0.2, 0.3]])
        EmbeddingsResponse(model='text-embedding-3-small', embeddings=[[0.1, 0.2, 0.3]], usage=None)
    """

    model: str
    embeddings: Sequence[Sequence[float]]
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """
    Catalog metadata for one model offered by a provider.

    Returned by ``BaseProvider.list_models``, one instance per model the
    vendor currently exposes. This is a lighter-weight, provider-adapter-local
    sibling of ``registry.models.ModelInfo`` — that other type is the
    gateway's curated, validated catalog entry (with capability enums,
    context window, etc.), whereas *this* ``ModelInfo`` is the raw,
    best-effort shape each adapter maps a vendor's own "list models" endpoint
    response into, often inferring boolean capability flags heuristically
    from the model's name (see e.g. ``ollama._map_models``, which infers
    ``supports_embeddings`` from whether ``"embed"`` appears in the model
    name, since Ollama's ``/api/tags`` response doesn't explicitly report
    capabilities).

    Attributes:
        id: The vendor's canonical identifier for the model, used when
            making subsequent API calls (e.g. Gemini's ``"models/gemini-pro"``
            path form). May differ from ``name`` for vendors whose "list"
            and "use" identifiers aren't identical.
        name: A human/API-friendly name for the model, e.g.
            ``"gemini-pro"`` (without a vendor-specific path prefix).
        display_name: An optional nicer label for UI display, or ``None``
            if the vendor doesn't provide one separately from ``name``.
        context_window: Maximum input token count the model accepts, if
            known; ``None`` otherwise. None of the current adapters populate
            this field (the underlying vendor "list models" endpoints used
            don't consistently report it), but it's modeled here for
            forward compatibility.
        supports_streaming: Whether the model can be used with
            ``stream_chat``. All four current adapters set this to ``True``
            unconditionally, since every model each currently-implemented
            vendor exposes supports streaming.
        supports_embeddings: Whether this model can be used for embeddings
            generation (as opposed to chat).
        supports_vision: Whether the model accepts image input.
        supports_tools: Whether the model supports function/tool calling.
        supports_json: Whether the model supports structured/JSON-constrained
            output.

    Example:
        >>> ModelInfo(
        ...     id="gpt-4o",
        ...     name="gpt-4o",
        ...     display_name="gpt-4o",
        ...     supports_streaming=True,
        ...     supports_vision=True,
        ...     supports_tools=True,
        ... )
        ModelInfo(id='gpt-4o', name='gpt-4o', display_name='gpt-4o', context_window=None, supports_streaming=True, supports_embeddings=False, supports_vision=True, supports_tools=True, supports_json=False)
    """

    id: str
    name: str
    display_name: str | None = None
    context_window: int | None = None
    supports_streaming: bool = False
    supports_embeddings: bool = False
    supports_vision: bool = False
    supports_tools: bool = False
    supports_json: bool = False


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """
    Result of a provider connectivity and readiness probe.

    Returned by ``BaseProvider.health_check``. Deliberately a *result*
    object rather than something that raises on failure: an "unhealthy"
    provider is an expected, normal outcome (e.g. a self-hosted Ollama
    instance that's temporarily down) that calling code — like a
    ``GET /health`` endpoint — needs to report gracefully, not treat as an
    exceptional program error.

    Attributes:
        healthy: Whether the provider responded successfully to the probe.
        latency_ms: How long the probe took, in milliseconds, whether it
            succeeded or failed — useful for monitoring/alerting on
            degraded-but-technically-healthy latency, or on how long a
            failed probe took to time out.
        message: A short human-readable summary of the result, e.g.
            ``"Ollama is reachable."`` or ``"Unable to reach Ollama."``.
            ``None`` if no message was generated.
        models_available: The number of models the provider reports having,
            if it could be determined from the probe response; ``None`` if
            unknown or the probe failed before that data was available.
        details: A free-form string-to-string bag of extra diagnostic
            context (e.g. ``{"status_code": "503"}`` or
            ``{"error": "ConnectTimeout"}``), defaulting to an empty dict so
            callers never need a ``None`` check before reading from it.

    Example:
        >>> HealthCheckResult(healthy=True, latency_ms=42.5, message="Ollama is reachable.", models_available=3)
        HealthCheckResult(healthy=True, latency_ms=42.5, message='Ollama is reachable.', models_available=3, details={})
        >>> HealthCheckResult(healthy=False, message="Unable to reach Ollama.", details={"error": "ConnectTimeout"})
        HealthCheckResult(healthy=False, latency_ms=None, message='Unable to reach Ollama.', models_available=None, details={'error': 'ConnectTimeout'})
    """

    healthy: bool
    latency_ms: float | None = None
    message: str | None = None
    models_available: int | None = None
    details: dict[str, str] = field(default_factory=dict)


class BaseProvider(ABC):
    """
    Abstract adapter for an external LLM provider.

    Each concrete implementation (Ollama, OpenAI, Anthropic, etc.) encapsulates
    vendor SDK/HTTP details and exposes a uniform async interface to the gateway
    service layer. Implementations must raise :class:`ProviderError` subclasses
    rather than leaking raw vendor exceptions.

    This is the contract the rest of the gateway (services, routes) codes
    against: as long as a class implements every abstract method here and is
    registered via ``register_provider`` (see ``registry.py``), it can be
    used anywhere a provider is expected, with no other code changes —
    exactly the extensibility goal described in
    ``05-ai-gateway-architecture.mdc``.

    Example:
        A minimal (non-functional) concrete implementation, showing the
        required shape:

        >>> class DemoProvider(BaseProvider):
        ...     provider_name = "demo"
        ...     async def chat(self, request):
        ...         return ChatResponse(content="demo response", model=request.model)
        ...     async def stream_chat(self, request):
        ...         yield ChatStreamChunk(content="demo")
        ...     async def embeddings(self, request):
        ...         return EmbeddingsResponse(model=request.model, embeddings=[[0.0]])
        ...     async def list_models(self):
        ...         return (ModelInfo(id="demo-model", name="demo-model"),)
        ...     async def health_check(self):
        ...         return HealthCheckResult(healthy=True)
        >>> import asyncio
        >>> provider = DemoProvider()
        >>> asyncio.run(provider.validate_model("demo-model"))
        True
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Stable provider identifier (e.g. ``ollama``, ``openai``).

        Declared as an abstract *property* (rather than a plain method)
        because it is treated as a constant, read-only piece of identity for
        the class — concrete adapters typically satisfy it with a simple
        class attribute (e.g. ``provider_name = "ollama"``), which Python
        allows to override an abstract property. This value is what
        :func:`~src.providers.registry.register_provider` uses as the
        registry key, and what every raised
        :class:`~src.providers.exceptions.ProviderError` records in its
        ``provider`` field.

        Returns:
            The provider's stable string identifier, matching the key it is
            (or will be) registered under in
            :class:`~src.providers.registry.ProviderRegistry`.
        """

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a non-streaming chat completion.

        Raises:
            ProviderError: On provider-side failure.

        Every concrete implementation additionally raises more specific
        :class:`~src.providers.exceptions.ProviderError` subclasses where
        applicable — e.g.
        :class:`~src.providers.exceptions.AuthenticationError`,
        :class:`~src.providers.exceptions.RateLimitError`,
        :class:`~src.providers.exceptions.InvalidRequestError`,
        :class:`~src.providers.exceptions.ModelNotFoundError`, or
        :class:`~src.providers.exceptions.ProviderUnavailableError` — via
        each HTTP-based adapter's ``HTTPErrorMapper`` (see
        ``http_errors.py``).

        Args:
            request: The normalized chat request to execute.

        Returns:
            The complete, normalized chat completion response.
        """

    @abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """
        Execute a streaming chat completion.

        Yields incremental content chunks until the response is complete.

        Raises:
            ProviderError: On provider-side failure.
            StreamingNotSupportedError: When streaming is unavailable.

        As with :meth:`chat`, concrete adapters may raise any
        :class:`~src.providers.exceptions.ProviderError` subclass that
        applies to the specific failure encountered, both before the stream
        begins (e.g. an invalid request rejected immediately) and mid-stream
        (e.g. a connection drop partway through — surfaced as
        :class:`~src.providers.exceptions.ProviderUnavailableError`).

        Args:
            request: The normalized chat request to execute, with
                ``request.stream`` typically (but not necessarily) set to
                ``True``.

        Returns:
            An async iterator of :class:`ChatStreamChunk` objects, yielded as
            the vendor produces them. The iterator is exhausted once the
            model finishes generating (typically signaled by a final chunk
            with ``finish_reason`` set).
        """

    @abstractmethod
    async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """
        Generate vector embeddings for the given input.

        Raises:
            ProviderError: On provider-side failure.
            UnsupportedCapabilityError: When the provider has no embeddings
                API at all (e.g. ``AnthropicProvider``, whose Messages API
                does not expose embeddings).

        Args:
            request: The normalized embeddings request to execute.

        Returns:
            The normalized embeddings response, with one vector per input.
        """

    @abstractmethod
    async def list_models(self) -> Sequence[ModelInfo]:
        """
        Return models currently available from this provider.

        Raises:
            ProviderError: When the catalog cannot be retrieved.

        Args:
            (none)

        Returns:
            A sequence of :class:`ModelInfo` describing every model the
            provider currently reports as available. HTTP-based cloud
            adapters typically cache this result briefly (see
            ``validation.ModelListCache``) to avoid refetching the catalog on
            every call.
        """

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Probe provider connectivity and basic readiness.

        Must not raise for expected unhealthy states; encode them in the result.
        Unexpected failures may still raise :class:`ProviderError`.

        This asymmetry (compared to the other abstract methods, which raise
        freely on any failure) exists because health checks are specifically
        meant to be safe to call speculatively/repeatedly (e.g. from a
        monitoring loop or a ``GET /health`` route) without the caller
        needing to wrap every call in a try/except just to detect "the
        provider is down," which is the single most expected outcome of a
        health check in the first place.

        Args:
            (none)

        Returns:
            A :class:`HealthCheckResult` describing whether the provider is
            reachable and any available diagnostic details.
        """

    async def validate_model(self, model: str) -> bool:
        """
        Return whether ``model`` is offered by this provider.

        Default implementation shared by every adapter: it fetches the full
        model catalog via :meth:`list_models` and checks for a match against
        either a model's ``id`` or ``name`` field, since different vendors'
        catalogs may make either one the "natural" identifier a caller would
        pass in. Cloud adapters (``OpenAIProvider``, ``AnthropicProvider``,
        ``GeminiProvider``) override this method to check their model-list
        *cache* first (see ``validation.ModelListCache``) before falling back
        to a fresh :meth:`list_models` call, avoiding a full catalog fetch on
        every single validation check.

        Args:
            model: The model identifier to check, as a caller would supply
                it in a :class:`ChatRequest` (e.g. ``"gpt-4o"``).

        Returns:
            ``True`` if any model in :meth:`list_models`'s result has a
            matching ``id`` or ``name``; ``False`` otherwise.

        Raises:
            ProviderError: If :meth:`list_models` itself fails (e.g. the
                catalog cannot be retrieved).

        Example:
            >>> import asyncio
            >>> class DemoProvider(BaseProvider):
            ...     provider_name = "demo"
            ...     async def chat(self, request): ...
            ...     async def stream_chat(self, request): ...
            ...     async def embeddings(self, request): ...
            ...     async def list_models(self):
            ...         return (ModelInfo(id="demo-model", name="demo-model"),)
            ...     async def health_check(self): ...
            >>> provider = DemoProvider()
            >>> asyncio.run(provider.validate_model("demo-model"))
            True
            >>> asyncio.run(provider.validate_model("unknown-model"))
            False
        """
        models = await self.list_models()
        return any(item.id == model or item.name == model for item in models)

    async def estimate_tokens(self, model: str, text: str) -> int:
        """
        Estimate token count for ``text``.

        Raises:
            UnsupportedCapabilityError: When the provider does not support estimation.

        No concrete adapter in this codebase currently overrides this
        default implementation — accurate token counting generally requires
        the vendor's specific tokenizer (e.g. ``tiktoken`` for OpenAI), which
        hasn't been integrated yet. The import of
        :class:`~src.providers.exceptions.UnsupportedCapabilityError` is
        deliberately local to this method (rather than at module level)
        purely to avoid a module-level dependency from ``base.py`` onto
        ``exceptions.py`` for what is, today, a single rarely-hit code path;
        other methods/modules import it at the top instead since they use it
        unconditionally.

        Args:
            model: The model identifier estimation would be performed for
                (unused by this default implementation, but part of the
                signature so a real implementation can pick a
                model-appropriate tokenizer).
            text: The text to estimate a token count for (also unused by
                this default implementation).

        Returns:
            This default implementation never returns; it always raises.

        Example:
            >>> import asyncio
            >>> class DemoProvider(BaseProvider):
            ...     provider_name = "demo"
            ...     async def chat(self, request): ...
            ...     async def stream_chat(self, request): ...
            ...     async def embeddings(self, request): ...
            ...     async def list_models(self): return ()
            ...     async def health_check(self): ...
            >>> provider = DemoProvider()
            >>> asyncio.run(provider.estimate_tokens("demo-model", "hello"))
            Traceback (most recent call last):
                ...
            UnsupportedCapabilityError: 'demo' does not support token estimation.
        """
        from src.providers.exceptions import UnsupportedCapabilityError

        raise UnsupportedCapabilityError(
            f"{self.provider_name!r} does not support token estimation.",
            provider=self.provider_name,
        )
