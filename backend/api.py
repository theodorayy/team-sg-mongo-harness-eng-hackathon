"""HTTP API for harness execution and conversation memory retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime, timezone
import hmac
import json
import re
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, unquote
from wsgiref.simple_server import make_server

from backend.contracts import SourceRecord
from backend.harness import HarnessServices, Mode, run_step
from backend.memory.conversation import ConversationStore, IdempotencyConflict
from backend.memory.retrieval import (
    RetrievalLimits, RetrievalStore, RetrievedContext, TokenCounter, retrieve_context,
)

def run_endpoint(
    records: Sequence[SourceRecord],
    mode: Mode,
    services: HarnessServices,
    *,
    session_id: str = '311-demo',
) -> dict[str, Any]:
    events = run_step(records, mode, services, session_id=session_id)

    def json_value(value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_value(item) for item in value]
        return value

    return {
        'run_id': events[0].run_id if events else None,
        'simulated_at': events[0].simulated_at.isoformat() if events else None,
        'events': [json_value(asdict(event)) for event in events],
    }


MAX_REQUEST_BYTES = 1_000_000
MAX_TEXT_CHARS = 100_000
DEFAULT_MESSAGE_LIMIT = 20
MAX_MESSAGE_LIMIT = 100


class MemoryApiService:
    """Application service that joins short-term messages to graph retrieval."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        retrieval_store: RetrievalStore,
        *,
        clock: Callable[[], datetime] | None = None,
        token_counter: TokenCounter | None = None,
        sources_collection: Any | None = None,
        api_token: str | None = None,
    ) -> None:
        self.conversation_store = conversation_store
        self.retrieval_store = retrieval_store
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.token_counter = token_counter
        self.sources_collection = sources_collection
        self.api_token = api_token or None

    def search_raw_sources(self, query: str, limit: int = 200) -> dict[str, Any]:
        """Search source_records by text query and return a large slice."""
        if self.sources_collection is None:
            return {"text": "", "token_estimate": 0, "count": 0}
        terms = query.strip().split()
        if not terms:
            return {"text": "", "token_estimate": 0, "count": 0}
        regex = "|".join(re.escape(t) for t in terms)
        docs = list(self.sources_collection.find(
            {"text": {"$regex": regex, "$options": "i"}},
            {"_id": 0, "id": 1, "text": 1, "occurred_at": 1, "metadata": 1},
        ).limit(limit))
        return self._format_source_docs(docs)

    def get_raw_sources(self, source_ids: list[str], limit: int = 50) -> dict[str, Any]:
        if self.sources_collection is None:
            return {"sources": [], "text": "", "token_estimate": 0}
        ids = source_ids[:limit]
        docs = list(self.sources_collection.find(
            {"id": {"$in": ids}},
            {"_id": 0, "id": 1, "text": 1, "occurred_at": 1, "metadata": 1},
        ))
        return self._format_source_docs(docs)

    def _format_source_docs(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        lines = []
        for doc in docs:
            meta = doc.get("metadata") or {}
            parts = [f"[Record {doc.get('id', '?')}]"]
            parts.append(f"Type: {meta.get('complaint_type', 'Unknown')}")
            if meta.get("descriptor"):
                parts.append(f"Descriptor: {meta['descriptor']}")
            if meta.get("borough"):
                parts.append(f"Borough: {meta['borough']}")
            if meta.get("incident_zip"):
                parts.append(f"Zip: {meta['incident_zip']}")
            if doc.get("occurred_at"):
                occ = doc["occurred_at"]
                parts.append(f"Date: {occ.isoformat() if hasattr(occ, 'isoformat') else occ}")
            if meta.get("agency"):
                parts.append(f"Agency: {meta['agency']}")
            lines.append(" | ".join(parts))
        text = "\n".join(lines)
        token_estimate = len(text.split())
        return {"text": text, "token_estimate": token_estimate, "count": len(docs)}

    def create_turn(
        self, session_id: str, prompt: str, idempotency_key: str
    ) -> dict[str, Any]:
        _validate_id(session_id, "session_id")
        _validate_text(prompt, "prompt")
        _validate_idempotency_key(idempotency_key)
        message = self.conversation_store.create_user_turn(
            session_id, prompt, idempotency_key, _utc(self.clock())
        )
        return {
            "session_id": message.session_id,
            "turn_id": message.turn_id,
            "prompt_message": message.as_dict(),
        }

    def get_context(
        self,
        session_id: str,
        turn_id: str,
        limits: Mapping[str, Any] | None = None,
    ) -> RetrievedContext:
        _validate_id(session_id, "session_id")
        _validate_id(turn_id, "turn_id")
        turn = self.conversation_store.get_user_turn(session_id, turn_id)
        if turn is None:
            raise KeyError("turn not found")
        retrieval_limits = RetrievalLimits(**dict(limits or {}))
        return retrieve_context(
            turn.content,
            session_id,
            turn.created_at,
            retrieval_limits,
            store=self.retrieval_store,
            token_counter=self.token_counter,
        )

    def record_response(
        self,
        session_id: str,
        turn_id: str,
        response: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        _validate_id(session_id, "session_id")
        _validate_id(turn_id, "turn_id")
        _validate_text(response, "response")
        _validate_idempotency_key(idempotency_key)
        message = self.conversation_store.append_assistant_response(
            session_id,
            turn_id,
            response,
            idempotency_key,
            _utc(self.clock()),
        )
        return message.as_dict()

    def list_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        _validate_id(session_id, "session_id")
        if not 1 <= limit <= MAX_MESSAGE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_MESSAGE_LIMIT}")
        return [
            message.as_dict()
            for message in self.conversation_store.list_messages(session_id, limit)
        ]


def create_app(service: MemoryApiService) -> Callable[..., Any]:
    """Create a WSGI application exposing the memory API.

    Routes:
      GET /healthz
      POST /v1/sessions/{session_id}/turns
      POST /v1/sessions/{session_id}/turns/{turn_id}/context
      POST /v1/sessions/{session_id}/turns/{turn_id}/response
      GET  /v1/sessions/{session_id}/messages?limit=20
    """

    def application(environ: Mapping[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        try:
            method = str(environ.get("REQUEST_METHOD", "GET")).upper()
            path = str(environ.get("PATH_INFO", "/"))
            if path == "/healthz" and method == "GET":
                return _respond(start_response, 200, {"ok": True})
            if service.api_token and not hmac.compare_digest(
                str(environ.get("HTTP_AUTHORIZATION", "")),
                f"Bearer {service.api_token}",
            ):
                return _respond(start_response, 401, {"error": "unauthorized"})
            query = parse_qs(str(environ.get("QUERY_STRING", "")))
            result, status = _dispatch(service, method, path, query, environ)
            return _respond(start_response, status, result)
        except KeyError as exc:
            return _respond(start_response, 404, {"error": str(exc).strip("'")})
        except IdempotencyConflict as exc:
            return _respond(start_response, 409, {"error": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return _respond(start_response, 400, {"error": str(exc)})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return _respond(start_response, 500, {"error": f"internal server error: {exc}"})

    return application


def serve(service: MemoryApiService, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run a basic WSGI server; production deployments can mount ``create_app``."""
    with make_server(host, port, create_app(service)) as server:
        server.serve_forever()


def _dispatch(
    service: MemoryApiService,
    method: str,
    path: str,
    query: Mapping[str, list[str]],
    environ: Mapping[str, Any],
) -> tuple[Any, int]:
    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "turns":
        if method != "POST":
            return {"error": "method not allowed"}, 405
        body = _read_json_body(environ)
        turn = service.create_turn(
            parts[2], _required_string(body, "prompt"), _required_string(body, "idempotency_key")
        )
        return turn, 201

    if len(parts) == 6 and parts[:2] == ["v1", "sessions"] and parts[3] == "turns":
        session_id, turn_id, action = parts[2], parts[4], parts[5]
        if action == "context":
            if method != "POST":
                return {"error": "method not allowed"}, 405
            body = _read_json_body(environ, allow_empty=True)
            limits = body.get("retrieval_limits", {})
            if not isinstance(limits, dict):
                raise ValueError("retrieval_limits must be an object")
            context = service.get_context(session_id, turn_id, limits)
            return context.as_dict(), 200
        if action == "response":
            if method != "POST":
                return {"error": "method not allowed"}, 405
            body = _read_json_body(environ)
            message = service.record_response(
                session_id,
                turn_id,
                _required_string(body, "content"),
                _required_string(body, "idempotency_key"),
            )
            return {"message": message}, 201
        return {"error": "route not found"}, 404

    if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":
        if method != "GET":
            return {"error": "method not allowed"}, 405
        raw_limit = (query.get("limit") or [str(DEFAULT_MESSAGE_LIMIT)])[0]
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError("limit must be an integer") from exc
        return {"messages": service.list_messages(parts[2], limit)}, 200

    if len(parts) == 3 and parts[:2] == ["v1", "sources"] and parts[2] == "search":
        if method != "POST":
            return {"error": "method not allowed"}, 405
        body = _read_json_body(environ)
        q = body.get("query", "")
        if not isinstance(q, str) or not q.strip():
            raise ValueError("query must be a non-empty string")
        limit = body.get("limit", 200)
        if not isinstance(limit, int) or limit < 1:
            limit = 200
        return service.search_raw_sources(q, min(limit, 500)), 200

    if len(parts) == 2 and parts[0] == "v1" and parts[1] == "sources":
        if method != "POST":
            return {"error": "method not allowed"}, 405
        body = _read_json_body(environ)
        ids = body.get("source_ids", [])
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError("source_ids must be a list of strings")
        limit = body.get("limit", 50)
        if not isinstance(limit, int) or limit < 1:
            limit = 50
        return service.get_raw_sources(ids, limit), 200

    return {"error": "route not found"}, 404


def _read_json_body(environ: Mapping[str, Any], allow_empty: bool = False) -> dict[str, Any]:
    raw_length = environ.get("CONTENT_LENGTH", "0")
    try:
        length = int(raw_length or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Content-Length") from exc
    if length < 0 or length > MAX_REQUEST_BYTES:
        raise ValueError(f"request body must be at most {MAX_REQUEST_BYTES} bytes")
    if length == 0 and allow_empty:
        return {}
    if length == 0:
        raise ValueError("request body is required")
    stream = environ.get("wsgi.input")
    if stream is None:
        raise ValueError("request body is unavailable")
    raw_body = stream.read(length)
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _required_string(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _validate_text(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    if len(value) > MAX_TEXT_CHARS:
        raise ValueError(f"{field} must be at most {MAX_TEXT_CHARS} characters")


def _validate_id(value: str, field: str) -> None:
    if not value or len(value) > 200 or not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise ValueError(f"{field} must be 1-200 URL-safe characters")


def _validate_idempotency_key(value: str) -> None:
    _validate_id(value, "idempotency_key")


def _respond(
    start_response: Callable[..., Any], status: int, payload: Mapping[str, Any]
) -> list[bytes]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    phrases = {
        200: "OK",
        201: "Created",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        500: "Internal Server Error",
    }
    start_response(
        f"{status} {phrases.get(status, 'Error')}",
        [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))],
    )
    return [body]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
