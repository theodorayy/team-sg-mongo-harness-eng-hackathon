"""Vector-seeded, budgeted retrieval over the long-term memory graph.

The module is storage-agnostic. The Atlas adapter supplies vector-search hits,
node/edge/source reads, and retrieval-counter updates through ``RetrievalStore``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from backend.contracts import MemoryEdge, MemoryNode, RetrievedContext


Document = Mapping[str, Any]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class RetrievalLimits:
    """Deterministic bounds for one retrieval operation."""

    seed_limit: int = 5
    max_hops: int = 2
    max_nodes: int = 30
    max_edges: int = 60
    max_context_tokens: int = 2_000
    max_members_per_summary: int = 5

    def __post_init__(self) -> None:
        if self.seed_limit < 1:
            raise ValueError("seed_limit must be at least 1")
        if self.max_hops < 0:
            raise ValueError("max_hops cannot be negative")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if self.max_edges < 0:
            raise ValueError("max_edges cannot be negative")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1")
        if self.max_members_per_summary < 0:
            raise ValueError("max_members_per_summary cannot be negative")


class RetrievalStore(Protocol):
    """Narrow read/write surface required from the MongoDB memory adapter."""

    def search_memories(
        self, text: str, limit: int, filters: Mapping[str, Any]
    ) -> Sequence[Document]:
        """Return documents shaped as ``{"node": ..., "score": float}``."""

    def get_memory_nodes(self, node_ids: Sequence[str]) -> Sequence[Document]:
        """Fetch nodes by application ID."""

    def get_memory_edges(
        self, node_ids: Sequence[str], limit: int
    ) -> Sequence[Document]:
        """Fetch incident edges for either endpoint in ``node_ids``."""

    def get_source_records(self, source_ids: Sequence[str]) -> Sequence[Document]:
        """Fetch source records, including their ``available_at`` timestamp."""

    def mark_nodes_retrieved(
        self, node_ids: Sequence[str], retrieved_at: datetime
    ) -> None:
        """Increment retrieval counters for nodes actually returned in context."""


_FOLLOW_RELATIONS = frozenset({"related_to", "supports", "contradicts", "member_of"})
_MAX_INLINE_SOURCE_IDS = 5
_RELATION_FACTOR = {
    "supports": 1.0,
    "related_to": 0.9,
    "contradicts": 0.8,
    "member_of": 0.8,
}
_DETAIL_QUERY_WORDS = frozenset(
    {
        "cite",
        "cited",
        "evidence",
        "example",
        "examples",
        "exact",
        "individual",
        "incident",
        "incidents",
        "proof",
        "record",
        "records",
        "source",
        "sources",
        "specific",
        "when",
        "where",
        "which",
    }
)


def retrieve_context(
    query: str,
    session_id: str,
    as_of: datetime,
    limits: RetrievalLimits,
    *,
    store: RetrievalStore,
    token_counter: TokenCounter | None = None,
) -> RetrievedContext:
    """Retrieve relevant long-term context, keeping it safe for replay time.

    The vector index supplies up to ``seed_limit`` semantically relevant nodes.
    Explicit graph-edge reads then expand those seeds, subject to fixed hop,
    node, edge, and context-token budgets. ``session_id`` remains in the public
    contract for the harness and trace layer; long-term memory is shared across
    sessions unless a caller passes a scope filter in the future.
    """
    if not query.strip():
        return _empty_context()
    if not session_id.strip():
        raise ValueError("session_id must not be empty")

    replay_time = _utc(as_of)
    count_tokens = token_counter or estimate_token_count
    detail_requested = _requests_member_detail(query)
    search_filters = {
        "status": "active",
        "first_seen_at": {"$lte": replay_time},
        "last_seen_at": {"$lte": replay_time},
    }

    requested_seed_candidates = max(limits.seed_limit * 4, limits.seed_limit)
    candidate_seed_limit = min(requested_seed_candidates, 100)

    summary_filters = {**search_filters, "kind": "summary"}
    summary_hits = store.search_memories(query, min(limits.seed_limit, 10), summary_filters)
    raw_hits = store.search_memories(query, candidate_seed_limit, search_filters)

    seen_ids: set[str] = set()
    merged: list[Document] = []
    for hit in summary_hits:
        doc_id = hit.get("id") or _document_id(hit)
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(hit)
    for hit in raw_hits:
        doc_id = hit.get("id") or _document_id(hit)
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(hit)

    eligible_seeds = _load_available_hits(store, merged, replay_time)
    eligible_seeds = [
        (node, score * 1.5 if node.kind == "summary" else score)
        for node, score in eligible_seeds
    ]
    eligible_seeds.sort(key=lambda item: (-item[1], item[0].id))
    truncated = (
        candidate_seed_limit < requested_seed_candidates
        or len(eligible_seeds) > limits.seed_limit
    )
    seeds = eligible_seeds[: limits.seed_limit]

    if not seeds:
        return _empty_context()

    candidates: dict[str, tuple[MemoryNode, float, int]] = {}
    seed_ids: list[str] = []
    for node, score in seeds:
        if node.id in candidates:
            continue
        if len(candidates) >= limits.max_nodes:
            break
        candidates[node.id] = (node, score, 0)
        seed_ids.append(node.id)

    truncated = truncated or len(seeds) > len(candidates)
    edges_by_id: dict[str, MemoryEdge] = {}
    frontier: deque[tuple[str, float, int]] = deque(
        (node.id, score, 0) for node, score in seeds if node.id in candidates
    )

    if detail_requested and limits.max_members_per_summary and limits.max_hops > 0:
        summary_nodes = [node for node, _ in seeds if node.kind == "summary"]
        for summary in summary_nodes:
            if len(candidates) >= limits.max_nodes:
                truncated = True
                break
            member_additions, membership_edges, member_was_truncated = _load_summary_members(
                store,
                query,
                summary,
                replay_time,
                limits,
                candidates,
            )
            truncated = truncated or member_was_truncated
            for node, score in member_additions:
                if node.id in candidates or len(candidates) >= limits.max_nodes:
                    if len(candidates) >= limits.max_nodes:
                        truncated = True
                    continue
                candidates[node.id] = (node, score, 1)
                frontier.append((node.id, score, 1))
            for edge in membership_edges:
                if len(edges_by_id) >= limits.max_edges:
                    truncated = True
                    break
                edges_by_id[edge.id] = edge

    for depth in range(limits.max_hops):
        if not frontier or len(candidates) >= limits.max_nodes:
            if frontier:
                truncated = True
            break

        current = [entry for entry in frontier if entry[2] == depth]
        frontier = deque(entry for entry in frontier if entry[2] > depth)
        expandable_ids = [
            node_id
            for node_id, _, _ in current
            if candidates[node_id][0].kind != "entity"
        ]
        if not expandable_ids:
            continue

        remaining_edges = max(0, limits.max_edges - len(edges_by_id))
        if remaining_edges == 0:
            truncated = True
            break
        raw_edges = store.get_memory_edges(expandable_ids, remaining_edges)
        parsed_edges = _load_available_edges(
            store, _parse_edges(raw_edges), replay_time
        )
        parsed_edges.sort(key=lambda edge: (-edge.weight, edge.id))

        traversable: list[tuple[MemoryEdge, str, str, float]] = []
        parent_scores = {node_id: score for node_id, score, _ in current}
        for edge in parsed_edges:
            if edge.relation not in _FOLLOW_RELATIONS:
                continue
            if edge.relation == "member_of" and not detail_requested:
                continue
            if len(edges_by_id) >= limits.max_edges:
                truncated = True
                break
            if edge.source_id in parent_scores:
                parent_id, neighbor_id = edge.source_id, edge.target_id
            elif edge.target_id in parent_scores:
                parent_id, neighbor_id = edge.target_id, edge.source_id
            else:
                continue
            # Summary membership is expanded by the dedicated, capped helper
            # above. Letting the generic walk follow these same edges would
            # bypass max_members_per_summary and make its limit misleading.
            if (
                edge.relation == "member_of"
                and candidates[parent_id][0].kind == "summary"
            ):
                continue
            edges_by_id.setdefault(edge.id, edge)
            score = (
                parent_scores[parent_id]
                * _RELATION_FACTOR.get(edge.relation, 0.75)
                * (0.5 ** (depth + 1))
            )
            traversable.append((edge, parent_id, neighbor_id, score))

        new_candidates: dict[str, tuple[float, int]] = {}
        for _, _, neighbor_id, score in traversable:
            if neighbor_id in candidates:
                continue
            prior = new_candidates.get(neighbor_id)
            if prior is None or score > prior[0]:
                new_candidates[neighbor_id] = (score, depth + 1)

        if not new_candidates:
            continue

        ranked_ids = sorted(new_candidates, key=lambda node_id: (-new_candidates[node_id][0], node_id))
        raw_nodes = store.get_memory_nodes(ranked_ids)
        available_nodes = _load_available_nodes(store, raw_nodes, replay_time)
        valid_ranked_ids = [node_id for node_id in ranked_ids if node_id in available_nodes]
        available_slots = max(0, limits.max_nodes - len(candidates))
        if len(valid_ranked_ids) > available_slots:
            valid_ranked_ids = valid_ranked_ids[:available_slots]
            truncated = True
        for node_id in valid_ranked_ids:
            node = available_nodes.get(node_id)
            if node is None:
                continue
            score, candidate_depth = new_candidates[node_id]
            candidates[node_id] = (node, score, candidate_depth)
            if node.kind != "entity" and candidate_depth < limits.max_hops:
                frontier.append((node_id, score, candidate_depth))

    selected_nodes = sorted(
        candidates.values(), key=lambda item: (item[2], -item[1], item[0].id)
    )
    selected_nodes, node_texts, token_count, token_truncated = _pack_nodes(
        selected_nodes, limits.max_nodes, limits.max_context_tokens, count_tokens
    )
    truncated = truncated or token_truncated or len(selected_nodes) < len(candidates)
    selected_ids = {node.id for node, _, _ in selected_nodes}
    selected_edges = _pack_edges(
        edges_by_id.values(), selected_ids, limits.max_edges
    )
    context_text, token_count, edge_truncated, rendered_edge_ids = _render_context(
        selected_nodes,
        selected_edges,
        limits.max_context_tokens,
        count_tokens,
        node_texts,
    )
    truncated = truncated or edge_truncated
    selected_edges = [edge for edge in selected_edges if edge.id in rendered_edge_ids]

    actual_seed_ids = [node_id for node_id in seed_ids if node_id in selected_ids]
    source_ids = sorted(
        {source_id for node, _, _ in selected_nodes for source_id in node.source_ids}
        | {source_id for edge in selected_edges for source_id in edge.source_ids}
    )
    if selected_nodes:
        store.mark_nodes_retrieved(
            [node.id for node, _, _ in selected_nodes], retrieved_at=replay_time
        )

    return RetrievedContext(
        seed_ids=actual_seed_ids,
        nodes=[node for node, _, _ in selected_nodes],
        edges=selected_edges,
        source_ids=source_ids,
        context_text=context_text,
        token_count=token_count,
        truncated=truncated,
    )


def estimate_token_count(text: str) -> int:
    """Deterministic estimate when a model tokenizer is absent.

    Pass the harness' model-compatible tokenizer through ``token_counter`` for
    exact accounting. The byte estimate is deliberately conservative for the
    prototype and keeps the cap deterministic without adding a dependency.
    """
    if not text:
        return 0
    return math.ceil(len(text.encode("utf-8")) / 3)


def _load_summary_members(
    store: RetrievalStore,
    query: str,
    summary: MemoryNode,
    as_of: datetime,
    limits: RetrievalLimits,
    selected: Mapping[str, tuple[MemoryNode, float, int]],
) -> tuple[list[tuple[MemoryNode, float]], list[MemoryEdge], bool]:
    if summary.group_id is not None:
        return [], [], False

    raw_edges = store.get_memory_edges([summary.id], limits.max_edges)
    membership_edges = _load_available_edges(
        store,
        [
            edge
            for edge in _parse_edges(raw_edges)
            if edge.relation == "member_of"
            and summary.id in {edge.source_id, edge.target_id}
        ],
        as_of,
    )
    member_ids = sorted(
        {
            edge.target_id if edge.source_id == summary.id else edge.source_id
            for edge in membership_edges
        }
        - set(selected)
    )
    if not member_ids:
        return [], [], False

    requested_member_candidates = min(
        len(member_ids), max(limits.max_members_per_summary * 4, limits.max_members_per_summary)
    )
    member_candidate_limit = min(requested_member_candidates, 100)
    allowed_ids = set(member_ids)
    try:
        member_hits = store.search_memories(
            query,
            member_candidate_limit,
            {
                "status": "active",
                "group_id": summary.id,
                "first_seen_at": {"$lte": as_of},
                "last_seen_at": {"$lte": as_of},
            },
        )
        ranked_hits: list[tuple[MemoryNode, float]] = []
        for document in member_hits:
            raw_node, score = _hit_parts(document)
            node = _parse_node(raw_node)
            if node.id in allowed_ids:
                ranked_hits.append((node, score))
    except Exception:
        member_docs = store.get_memory_nodes(list(allowed_ids)[:member_candidate_limit])
        ranked_hits = [
            (_parse_node(doc), 0.5)
            for doc in member_docs
            if _parse_node(doc).status == "active"
        ]
    available_nodes = _load_available_nodes(
        store, [node.as_dict() for node, _ in ranked_hits], as_of
    )
    edge_by_member = {
        edge.target_id if edge.source_id == summary.id else edge.source_id: edge
        for edge in membership_edges
    }
    additions = [
        (available_nodes[node.id], score)
        for node, score in ranked_hits
        if node.id in available_nodes
    ][: limits.max_members_per_summary]
    chosen_ids = {node.id for node, _ in additions}
    chosen_edges = [edge_by_member[node_id] for node_id in sorted(chosen_ids) if node_id in edge_by_member]
    truncated = (
        len(available_nodes) > len(additions)
        or member_candidate_limit < len(member_ids)
    )
    return additions, chosen_edges, truncated


def _load_available_hits(
    store: RetrievalStore, hits: Sequence[Document], as_of: datetime
) -> list[tuple[MemoryNode, float]]:
    parsed: list[tuple[MemoryNode, float]] = []
    for hit in hits:
        raw_node, score = _hit_parts(hit)
        parsed.append((_parse_node(raw_node), score))
    available = _load_available_nodes(
        store, [node.as_dict() for node, _ in parsed], as_of
    )
    return [(available[node.id], score) for node, score in parsed if node.id in available]


def _load_available_nodes(
    store: RetrievalStore, documents: Sequence[Document], as_of: datetime
) -> dict[str, MemoryNode]:
    parsed_nodes = [_parse_node(document) for document in documents]
    source_ids = sorted({source_id for node in parsed_nodes for source_id in node.source_ids})
    source_docs = store.get_source_records(source_ids) if source_ids else []
    availability = {
        _document_id(source): _datetime_value(source.get("available_at"))
        for source in source_docs
    }

    available: dict[str, MemoryNode] = {}
    for node in parsed_nodes:
        if node.status != "active":
            continue
        if node.first_seen_at and _utc(node.first_seen_at) > as_of:
            continue
        if node.last_seen_at and _utc(node.last_seen_at) > as_of:
            continue
        # A summary that mixes in future evidence is excluded whole; trimming
        # its source IDs alone would leave future-derived text in the prompt.
        if not node.source_ids:
            continue
        if any(
            source_id not in availability
            or availability[source_id] is None
            or _utc(availability[source_id]) > as_of
            for source_id in node.source_ids
        ):
            continue
        available[node.id] = node
    return available


def _load_available_edges(
    store: RetrievalStore, edges: Sequence[MemoryEdge], as_of: datetime
) -> list[MemoryEdge]:
    source_ids = sorted({source_id for edge in edges for source_id in edge.source_ids})
    if not source_ids:
        return []
    source_docs = store.get_source_records(source_ids)
    availability = {
        _document_id(source): _datetime_value(source.get("available_at"))
        for source in source_docs
    }
    return [
        edge
        for edge in edges
        if edge.source_ids
        and all(
            source_id in availability
            and availability[source_id] is not None
            and _utc(availability[source_id]) <= as_of
            for source_id in edge.source_ids
        )
    ]


def _parse_node(document: Document) -> MemoryNode:
    return MemoryNode(
        id=_document_id(document),
        kind=str(document.get("kind", "fact")),
        text=str(document.get("text", "")),
        scope_key=str(document.get("scope_key", "")),
        source_ids=sorted({str(item) for item in document.get("source_ids", [])}),
        first_seen_at=_datetime_value(document.get("first_seen_at")),
        last_seen_at=_datetime_value(document.get("last_seen_at")),
        status=str(document.get("status", "active")),
        group_id=(str(document["group_id"]) if document.get("group_id") is not None else None),
    )


def _parse_edges(documents: Sequence[Document]) -> list[MemoryEdge]:
    edges: dict[str, MemoryEdge] = {}
    for document in documents:
        source_id = str(document.get("source_id", ""))
        target_id = str(document.get("target_id", ""))
        if not source_id or not target_id:
            continue
        edge = MemoryEdge(
            id=_document_id(document),
            source_id=source_id,
            target_id=target_id,
            relation=str(document.get("relation", "")),
            weight=float(document.get("weight", 1.0)),
            source_ids=sorted({str(item) for item in document.get("source_ids", [])}),
        )
        edges[edge.id] = edge
    return list(edges.values())


def _hit_parts(hit: Document) -> tuple[Document, float]:
    nested_node = hit.get("node", hit.get("memory"))
    node = nested_node if isinstance(nested_node, Mapping) else hit
    score_value = hit.get("score", hit.get("vector_score", hit.get("similarity", 0.0)))
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        score = 0.0
    return node, score


def _document_id(document: Document) -> str:
    value = document.get("id", document.get("_id"))
    if value is None:
        raise ValueError("memory/source document is missing its id")
    return str(value)


def _datetime_value(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _utc(parsed)
    raise TypeError(f"expected datetime or ISO timestamp, got {type(value).__name__}")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _requests_member_detail(query: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", query.casefold()))
    return bool(words & _DETAIL_QUERY_WORDS)


def _pack_nodes(
    ranked_nodes: Sequence[tuple[MemoryNode, float, int]],
    max_nodes: int,
    max_tokens: int,
    token_counter: TokenCounter,
) -> tuple[list[tuple[MemoryNode, float, int]], dict[str, str], int, bool]:
    selected: list[tuple[MemoryNode, float, int]] = []
    rendered: dict[str, str] = {}
    truncated = len(ranked_nodes) > max_nodes
    for node, score, depth in ranked_nodes:
        if len(selected) >= max_nodes:
            truncated = True
            break
        fragment = _render_node(node, score, depth)
        node_body = "\n\n".join([*rendered.values(), fragment])
        tentative = _context_prefix() + "\n\n" + node_body
        if token_counter(tentative) > max_tokens:
            truncated = True
            continue
        selected.append((node, score, depth))
        rendered[node.id] = fragment
    text = (
        _context_prefix() + "\n\n" + "\n\n".join(rendered.values())
        if rendered
        else ""
    )
    return selected, rendered, token_counter(text), truncated


def _pack_edges(
    edges: Sequence[MemoryEdge],
    selected_ids: set[str],
    max_edges: int,
) -> list[MemoryEdge]:
    candidates = [
        edge
        for edge in edges
        if edge.source_id in selected_ids and edge.target_id in selected_ids
    ]
    candidates.sort(key=lambda edge: (-edge.weight, edge.relation, edge.id))
    return candidates[:max_edges]


def _render_node(node: MemoryNode, score: float, depth: int) -> str:
    src_count = len(node.source_ids)
    suffix = f" [{src_count} sources]" if src_count else ""
    return f"- {node.text.strip()}{suffix}"


def _render_context(
    ranked_nodes: Sequence[tuple[MemoryNode, float, int]],
    edges: Sequence[MemoryEdge],
    max_tokens: int,
    token_counter: TokenCounter,
    node_texts: Mapping[str, str],
) -> tuple[str, int, bool, list[str]]:
    if not ranked_nodes:
        return "", 0, False, []
    node_ids = {node.id for node, _, _ in ranked_nodes}
    lines = [_context_prefix()]
    lines.extend(node_texts[node.id] for node, _, _ in ranked_nodes)
    edge_line_by_id: dict[str, str] = {}
    if edges:
        edge_line_by_id = {
            edge.id: f"- {edge.relation}: {edge.source_id} → {edge.target_id}"
            for edge in edges
            if edge.source_id in node_ids and edge.target_id in node_ids
        }
        edge_lines = list(edge_line_by_id.values())
        if edge_lines:
            lines.append("Relevant relationships:")
            lines.extend(edge_lines)

    context = "\n".join(lines)
    if token_counter(context) <= max_tokens:
        return context, token_counter(context), False, list(edge_line_by_id)

    # Node packing already reserves the main budget. If header/edge text pushes
    # the final rendering over it, shed lower-ranked edges deterministically.
    edge_lines = [line for line in lines if line.startswith("[Graph link]")]
    base_lines = [line for line in lines if not line.startswith("[Graph link]")]
    if "Relevant relationships:" in base_lines:
        base_lines.remove("Relevant relationships:")
    truncated = bool(edge_lines)
    while edge_lines:
        candidate = "\n".join([*base_lines, "Relevant relationships:", *edge_lines])
        if token_counter(candidate) <= max_tokens:
            included_lines = set(edge_lines)
            included_ids = [
                edge_id
                for edge_id, edge_line in edge_line_by_id.items()
                if edge_line in included_lines
            ]
            return candidate, token_counter(candidate), truncated, included_ids
        edge_lines.pop()
    base_context = "\n".join(base_lines)
    return base_context, token_counter(base_context), True, []


def _empty_context() -> RetrievedContext:
    return RetrievedContext(
        seed_ids=[],
        nodes=[],
        edges=[],
        source_ids=[],
        context_text="",
        token_count=0,
        truncated=False,
    )


def _context_prefix() -> str:
    return "Knowledge graph patterns:"
