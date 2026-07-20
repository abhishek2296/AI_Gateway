# AI Gateway Project Memory

## Vision

Build an AI Gateway, not a coding assistant.

## Workflow

Theory → Architecture → Cursor Implementation → Code Review → Refactor →
Test

## Completed

-   Phase 1
-   Phase 2
-   Phase 3.1 Database Infrastructure
-   Phase 3.2 ORM Foundation

## Next Domain Models

1.  Provider
2.  Model
3.  ChatSession
4.  Message
5.  PromptTemplate
6.  APIKey
7.  ProviderConfiguration

## Architecture Rules

-   Single SQLAlchemy metadata registry
-   SQLAlchemy 2.x typed ORM
-   Provider-agnostic architecture
-   Review before implementation
-   Do not rewrite existing code without reason

## Future Stack

Development: - PostgreSQL - Ollama

Later: - Redis - Qdrant - Prometheus - Grafana - Nginx

## ADRs

docs/architecture/ 001-single-metadata-registry.md
002-timestamp-mixin.md 003-provider-abstraction.md 004-model-registry.md
