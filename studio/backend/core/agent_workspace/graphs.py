# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Project-scoped durable sequential graphs.

Graphs are orchestration records. They do not contain a second model or agent
runtime. Loop and model nodes submit work to the existing background-agent
manager, while tool nodes use the existing MCP transport.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from string import Formatter
from typing import Any, Callable, Optional

from core.inference.mcp_client import call_tool_sync, parse_server_headers
from storage import mcp_servers_db

from .common import AgentWorkspaceError, now_ms
from .state import connection


_GRAPH_STATUSES = frozenset(
    {
        "queued",
        "running",
        "pausing",
        "paused",
        "cancelling",
        "cancelled",
        "completed",
        "failed",
        "interrupted",
    }
)
_NODE_STATUSES = frozenset({"running", "paused", "cancelled", "completed", "failed", "interrupted"})
_APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected"})
_NODE_TYPES = frozenset({"input", "loop", "model", "tool", "condition", "approval", "output"})
_MAX_NODES = 100
_MAX_EDGES = 200
_MAX_JSON_BYTES = 256 * 1024
_MAX_GRAPH_DOCUMENT_BYTES = 512 * 1024
_MAX_RUN_OUTPUT_BYTES = 1024 * 1024
_MAX_RUN_SECONDS = 24 * 60 * 60
_MAX_NODE_SECONDS = 2 * 60 * 60
_UNSET = object()


def _json(value: Any, *, limit: int, label: str) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii = False, separators = (",", ":"))
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} must be JSON serializable.") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise AgentWorkspaceError(f"{label} is too large.")
    return encoded


def _load(value: Optional[str], default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _string(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
    maximum: int = 512,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return value.strip()


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise AgentWorkspaceError(f"{label} is invalid.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} is invalid.") from exc
    if result < minimum or result > maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return result


def _validate_runtime(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise AgentWorkspaceError(f"{label} must be an object.")
    allowed = {
        "kind",
        "model",
        "providerId",
        "permissionMode",
        "reasoningEffort",
        "maxOutputTokens",
    }
    if set(value) - allowed:
        raise AgentWorkspaceError(f"{label} contains unsupported fields.")
    kind = value.get("kind")
    if kind not in {"local", "provider"}:
        raise AgentWorkspaceError(f"{label}.kind is invalid.")
    model = _string(value.get("model"), f"{label}.model", maximum = 512)
    permission = value.get("permissionMode")
    if permission not in {"off", "full"}:
        raise AgentWorkspaceError(f"{label}.permissionMode is invalid.")
    result = {
        "kind": kind,
        "model": model,
        "permissionMode": permission,
        "maxOutputTokens": _bounded_int(
            value.get("maxOutputTokens", 8192), f"{label}.maxOutputTokens", 1, 32768
        ),
    }
    for key in ("providerId", "reasoningEffort"):
        if value.get(key) is not None:
            result[key] = _string(value[key], f"{label}.{key}", maximum = 256)
    if kind == "provider" and not result.get("providerId"):
        raise AgentWorkspaceError(f"{label}.providerId is required for provider runtimes.")
    return result


def _validate_node(node: Any) -> dict:
    if not isinstance(node, dict) or set(node) - {"id", "type", "config", "label"}:
        raise AgentWorkspaceError("Graph node is invalid.")
    node_id = _string(node.get("id"), "Graph node ID", maximum = 128)
    node_type = node.get("type")
    if node_type not in _NODE_TYPES:
        raise AgentWorkspaceError(f"Graph node '{node_id}' has an invalid type.")
    config = node.get("config") or {}
    if not isinstance(config, dict):
        raise AgentWorkspaceError(f"Graph node '{node_id}' config must be an object.")
    allowed: set[str]
    normalized: dict[str, Any] = {}
    if node_type == "input":
        allowed = {"name"}
        normalized["name"] = _string(config.get("name", "input"), "Input name", maximum = 128)
    elif node_type in {"loop", "model"}:
        allowed = {"instruction", "prompt", "runtime", "timeoutSeconds"}
        text_key = "instruction" if node_type == "loop" else "prompt"
        normalized[text_key] = _string(config.get(text_key), f"{node_type} prompt", maximum = 32768)
        if config.get("runtime") is not None:
            normalized["runtime"] = _validate_runtime(config["runtime"], f"{node_type} runtime")
        normalized["timeoutSeconds"] = _bounded_int(
            config.get("timeoutSeconds", _MAX_NODE_SECONDS),
            f"{node_type}.timeoutSeconds",
            1,
            _MAX_NODE_SECONDS,
        )
    elif node_type == "tool":
        allowed = {"serverId", "toolName", "arguments", "timeoutSeconds"}
        normalized["serverId"] = _string(config.get("serverId"), "Tool serverId", maximum = 128)
        normalized["toolName"] = _string(config.get("toolName"), "Tool name", maximum = 256)
        arguments = config.get("arguments", {})
        if not isinstance(arguments, dict):
            raise AgentWorkspaceError("Tool arguments must be an object.")
        _json(arguments, limit = 64 * 1024, label = "Tool arguments")
        normalized["arguments"] = arguments
        normalized["timeoutSeconds"] = _bounded_int(
            config.get("timeoutSeconds", 300), "Tool timeoutSeconds", 1, _MAX_NODE_SECONDS
        )
    elif node_type == "condition":
        allowed = {"path", "operator", "value"}
        normalized["path"] = _string(config.get("path"), "Condition path", maximum = 512)
        operator = config.get("operator", "truthy")
        if operator not in {"truthy", "falsy", "exists", "equals", "notEquals"}:
            raise AgentWorkspaceError("Condition operator is invalid.")
        normalized["operator"] = operator
        if operator in {"equals", "notEquals"} and "value" not in config:
            raise AgentWorkspaceError("Condition value is required for equality operators.")
        if "value" in config:
            _json(config["value"], limit = 32 * 1024, label = "Condition value")
            normalized["value"] = config["value"]
    elif node_type == "approval":
        allowed = {"title", "description"}
        normalized["title"] = _string(
            config.get("title", "Approval required"), "Approval title", maximum = 500
        )
        normalized["description"] = str(config.get("description", ""))[:4000]
    else:
        allowed = {"name", "path"}
        normalized["name"] = _string(config.get("name", node_id), "Output name", maximum = 128)
        if config.get("path") is not None:
            normalized["path"] = _string(config["path"], "Output path", maximum = 512)
    if set(config) - allowed:
        raise AgentWorkspaceError(f"Graph node '{node_id}' config contains unsupported fields.")
    result = {"id": node_id, "type": node_type, "config": normalized}
    if node.get("label") is not None:
        result["label"] = _string(node["label"], "Graph node label", maximum = 200)
    return result


def validate_graph_spec(spec: dict) -> dict:
    """Validate and canonicalize one immutable graph revision."""
    if not isinstance(spec, dict):
        raise AgentWorkspaceError("Graph definition must be an object.")
    allowed = {
        "name",
        "description",
        "metadata",
        "inputSchema",
        "outputSchema",
        "nodes",
        "edges",
        "permissions",
        "limits",
    }
    if set(spec) - allowed:
        raise AgentWorkspaceError("Graph definition contains unsupported fields.")
    name = _string(spec.get("name"), "Graph name", maximum = 200)
    description = str(spec.get("description", ""))[:4000]
    metadata = spec.get("metadata", {})
    if not isinstance(metadata, dict):
        raise AgentWorkspaceError("Graph metadata must be an object.")
    _json(metadata, limit = 64 * 1024, label = "Graph metadata")
    input_schema = spec.get("inputSchema", {"type": "object"})
    output_schema = spec.get("outputSchema", {"type": "object"})
    if not isinstance(input_schema, dict) or not isinstance(output_schema, dict):
        raise AgentWorkspaceError("Graph schemas must be objects.")
    _validate_schema_definition(input_schema, "Graph input")
    _validate_schema_definition(output_schema, "Graph output")
    _json(input_schema, limit = 64 * 1024, label = "Graph input schema")
    _json(output_schema, limit = 64 * 1024, label = "Graph output schema")
    raw_nodes = spec.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes or len(raw_nodes) > _MAX_NODES:
        raise AgentWorkspaceError(f"Graph must contain 1 to {_MAX_NODES} nodes.")
    nodes = [_validate_node(node) for node in raw_nodes]
    node_ids = [node["id"] for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise AgentWorkspaceError("Graph node IDs must be unique.")
    by_id = set(node_ids)
    raw_edges = spec.get("edges", [])
    if not isinstance(raw_edges, list) or len(raw_edges) > _MAX_EDGES:
        raise AgentWorkspaceError(f"Graph must contain at most {_MAX_EDGES} edges.")
    edges = []
    incoming = {node_id: 0 for node_id in by_id}
    outgoing: dict[str, list[dict]] = {node_id: [] for node_id in by_id}
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict) or set(raw_edge) - {"from", "to", "when"}:
            raise AgentWorkspaceError("Graph edge is invalid.")
        source = _string(raw_edge.get("from"), "Graph edge source", maximum = 128)
        target = _string(raw_edge.get("to"), "Graph edge target", maximum = 128)
        if source not in by_id or target not in by_id:
            raise AgentWorkspaceError("Graph edges must reference existing nodes.")
        if source == target:
            raise AgentWorkspaceError("Graph cannot contain self edges.")
        when = raw_edge.get("when")
        if when is not None and when not in {"true", "false", "default"}:
            raise AgentWorkspaceError("Graph edge condition is invalid.")
        source_type = next(node["type"] for node in nodes if node["id"] == source)
        if source_type != "condition" and when is not None:
            raise AgentWorkspaceError("Only condition nodes may have conditional edges.")
        edge = {"from": source, "to": target}
        if when is not None:
            edge["when"] = when
        edges.append(edge)
        incoming[target] += 1
        outgoing[source].append(edge)
    inputs = [node for node in nodes if node["type"] == "input"]
    if len(inputs) != 1 or incoming[inputs[0]["id"]] != 0:
        raise AgentWorkspaceError("Graph must have exactly one root input node.")
    if any(count > 1 for count in incoming.values()):
        raise AgentWorkspaceError("Graph joins are not supported in the sequential graph version.")
    for node in nodes:
        node_edges = outgoing[node["id"]]
        if node["type"] == "condition":
            if len(node_edges) not in {1, 2}:
                raise AgentWorkspaceError("Condition nodes need one or two outgoing edges.")
            conditions = [edge.get("when") for edge in node_edges]
            if len(node_edges) == 1 and conditions[0] not in {None, "default"}:
                raise AgentWorkspaceError(
                    "A single condition edge must be unconditional or default."
                )
            if len(node_edges) == 2 and set(conditions) != {"true", "false"}:
                raise AgentWorkspaceError("Two condition edges must be true and false.")
        elif len(node_edges) > 1:
            raise AgentWorkspaceError(f"Node '{node['id']}' has too many outgoing edges.")
    terminals = [node for node in nodes if not outgoing[node["id"]]]
    if not terminals or not any(node["type"] == "output" for node in terminals):
        raise AgentWorkspaceError("Graph must terminate at an output node.")
    visited: set[str] = set()
    stack = [inputs[0]["id"]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(edge["to"] for edge in outgoing[current])
    if visited != by_id:
        raise AgentWorkspaceError("Graph contains unreachable nodes.")
    indegrees = dict(incoming)
    queue = [node_id for node_id, count in indegrees.items() if count == 0]
    visited_count = 0
    while queue:
        current = queue.pop()
        visited_count += 1
        for edge in outgoing[current]:
            indegrees[edge["to"]] -= 1
            if indegrees[edge["to"]] == 0:
                queue.append(edge["to"])
    if visited_count != len(nodes):
        raise AgentWorkspaceError("Graph cannot contain cycles.")
    permissions = spec.get("permissions", {})
    if not isinstance(permissions, dict) or set(permissions) - {"allowedToolServerIds"}:
        raise AgentWorkspaceError("Graph permissions are invalid.")
    allowed_tools = permissions.get("allowedToolServerIds", [])
    if not isinstance(allowed_tools, list) or any(
        not isinstance(item, str) for item in allowed_tools
    ):
        raise AgentWorkspaceError("Graph allowedToolServerIds is invalid.")
    limits = spec.get("limits", {})
    if not isinstance(limits, dict) or set(limits) - {
        "maxNodes",
        "maxRunSeconds",
        "maxOutputBytes",
    }:
        raise AgentWorkspaceError("Graph limits are invalid.")
    normalized_limits = {
        "maxNodes": _bounded_int(
            limits.get("maxNodes", len(nodes)), "Graph maxNodes", 1, _MAX_NODES
        ),
        "maxRunSeconds": _bounded_int(
            limits.get("maxRunSeconds", 3600), "Graph maxRunSeconds", 1, _MAX_RUN_SECONDS
        ),
        "maxOutputBytes": _bounded_int(
            limits.get("maxOutputBytes", _MAX_RUN_OUTPUT_BYTES),
            "Graph maxOutputBytes",
            1024,
            _MAX_RUN_OUTPUT_BYTES,
        ),
    }
    if normalized_limits["maxNodes"] < len(nodes):
        raise AgentWorkspaceError("Graph maxNodes cannot be lower than the node count.")
    result = {
        "name": name,
        "description": description,
        "metadata": metadata,
        "inputSchema": input_schema,
        "outputSchema": output_schema,
        "nodes": nodes,
        "edges": edges,
        "permissions": {"allowedToolServerIds": sorted(set(allowed_tools))},
        "limits": normalized_limits,
    }
    _json(result, limit = _MAX_GRAPH_DOCUMENT_BYTES, label = "Graph definition")
    return result


def _ensure_schema(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_graphs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            current_revision INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE(project_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graphs_project
            ON agent_graphs(project_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS agent_graph_revisions (
            graph_id TEXT NOT NULL REFERENCES agent_graphs(id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            document_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY(graph_id, revision)
        );
        CREATE TABLE IF NOT EXISTS agent_graph_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            graph_id TEXT NOT NULL REFERENCES agent_graphs(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT,
            error TEXT,
            current_node_id TEXT,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            retry_of_run_id TEXT REFERENCES agent_graph_runs(id) ON DELETE SET NULL,
            idempotency_key TEXT,
            pause_requested INTEGER NOT NULL DEFAULT 0,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_runs_project
            ON agent_graph_runs(project_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS agent_graph_node_executions (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT,
            output_json TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            UNIQUE(run_id, node_id, attempt)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_node_runs
            ON agent_graph_node_executions(run_id, created_at);
        CREATE TABLE IF NOT EXISTS agent_graph_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            node_id TEXT,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(run_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS agent_graph_approvals (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES agent_graph_runs(id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            decision TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE(run_id, node_id)
        );
        CREATE INDEX IF NOT EXISTS idx_agent_graph_approvals_project
            ON agent_graph_approvals(project_id, updated_at DESC);
        """
    )
    index_columns = [
        row[2]
        for row in conn.execute("PRAGMA index_info(idx_agent_graph_runs_idempotency)").fetchall()
    ]
    if index_columns != ["project_id", "graph_id", "idempotency_key"]:
        conn.execute("DROP INDEX IF EXISTS idx_agent_graph_runs_idempotency")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_graph_runs_idempotency "
            "ON agent_graph_runs(project_id, graph_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL"
        )
    conn.commit()


def _conn():
    conn = connection()
    _ensure_schema(conn)
    return conn


def _revision_document(row) -> dict:
    document = _load(row["document_json"], {})
    return {
        **document,
        "graphId": row["graph_id"],
        "projectId": row["project_id"],
        "revision": row["revision"],
        "createdAt": row["created_at"],
    }


def _graph(conn, row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "description": row["description"],
        "currentRevision": row["current_revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def create_graph(project_id: str, spec: dict) -> dict:
    document = validate_graph_spec(spec)
    graph_id = str(uuid.uuid4())
    current = now_ms()
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO agent_graphs(id, project_id, name, description, current_revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (graph_id, project_id, document["name"], document["description"], current, current),
        )
        conn.execute(
            "INSERT INTO agent_graph_revisions(graph_id, project_id, revision, document_json, created_at) "
            "VALUES (?, ?, 1, ?, ?)",
            (
                graph_id,
                project_id,
                _json(document, limit = _MAX_GRAPH_DOCUMENT_BYTES, label = "Graph definition"),
                current,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graphs WHERE id = ?", (graph_id,)).fetchone()
        return _graph(conn, row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_graph(project_id: str, graph_id: str, spec: dict, *, expected_revision: int) -> dict:
    document = validate_graph_spec(spec)
    current = now_ms()
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_graphs WHERE id = ? AND project_id = ?", (graph_id, project_id)
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph not found.")
        if row["current_revision"] != expected_revision:
            raise AgentWorkspaceError("Graph changed in another session. Refresh and retry.")
        revision = int(row["current_revision"]) + 1
        conn.execute(
            "INSERT INTO agent_graph_revisions(graph_id, project_id, revision, document_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                graph_id,
                project_id,
                revision,
                _json(document, limit = _MAX_GRAPH_DOCUMENT_BYTES, label = "Graph definition"),
                current,
            ),
        )
        conn.execute(
            "UPDATE agent_graphs SET name = ?, description = ?, current_revision = ?, updated_at = ? WHERE id = ?",
            (document["name"], document["description"], revision, current, graph_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM agent_graphs WHERE id = ?", (graph_id,)).fetchone()
        return _graph(conn, updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph(project_id: str, graph_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graphs WHERE id = ? AND project_id = ?", (graph_id, project_id)
        ).fetchone()
        return _graph(conn, row) if row else None
    finally:
        conn.close()


def list_graphs(project_id: str) -> list[dict]:
    conn = _conn()
    try:
        return [
            _graph(conn, row)
            for row in conn.execute(
                "SELECT * FROM agent_graphs WHERE project_id = ? ORDER BY updated_at DESC, id",
                (project_id,),
            ).fetchall()
        ]
    finally:
        conn.close()


def delete_graph(project_id: str, graph_id: str) -> None:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id FROM agent_graphs WHERE id = ? AND project_id = ?",
            (graph_id, project_id),
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph not found.")
        active = conn.execute(
            "SELECT 1 FROM agent_graph_runs WHERE graph_id = ? AND status IN "
            "('queued', 'running', 'pausing', 'paused', 'cancelling') LIMIT 1",
            (graph_id,),
        ).fetchone()
        if active is not None:
            raise AgentWorkspaceError("Stop active graph runs before deleting this graph.")
        conn.execute(
            "DELETE FROM agent_graphs WHERE id = ? AND project_id = ?", (graph_id, project_id)
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph_revision(
    project_id: str,
    graph_id: str,
    revision: Optional[int] = None,
) -> Optional[dict]:
    conn = _conn()
    try:
        if revision is None:
            row = conn.execute(
                "SELECT r.* FROM agent_graph_revisions r JOIN agent_graphs g ON g.id = r.graph_id "
                "WHERE r.project_id = ? AND r.graph_id = ? AND r.revision = g.current_revision",
                (project_id, graph_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM agent_graph_revisions WHERE project_id = ? AND graph_id = ? AND revision = ?",
                (project_id, graph_id, revision),
            ).fetchone()
        return _revision_document(row) if row else None
    finally:
        conn.close()


def _run(row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "graphId": row["graph_id"],
        "revision": row["revision"],
        "input": _load(row["input_json"], {}),
        "output": _load(row["output_json"], None),
        "error": row["error"],
        "currentNodeId": row["current_node_id"],
        "status": row["status"],
        "attempt": row["attempt"],
        "retryOfRunId": row["retry_of_run_id"],
        "idempotencyKey": row["idempotency_key"],
        "pauseRequested": bool(row["pause_requested"]),
        "cancelRequested": bool(row["cancel_requested"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def create_graph_run(
    project_id: str,
    graph_id: str,
    input_data: Any,
    *,
    revision: Optional[int] = None,
    idempotency_key: Optional[str] = None,
    retry_of_run_id: Optional[str] = None,
) -> dict:
    if not isinstance(input_data, dict):
        raise AgentWorkspaceError("Graph run input must be an object.")
    _json(input_data, limit = _MAX_JSON_BYTES, label = "Graph run input")
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        graph = conn.execute(
            "SELECT current_revision FROM agent_graphs WHERE id = ? AND project_id = ?",
            (graph_id, project_id),
        ).fetchone()
        if graph is None:
            raise AgentWorkspaceError("Graph not found.")
        selected_revision = int(graph["current_revision"] if revision is None else revision)
        document_row = conn.execute(
            "SELECT document_json FROM agent_graph_revisions WHERE graph_id = ? AND project_id = ? AND revision = ?",
            (graph_id, project_id, selected_revision),
        ).fetchone()
        if document_row is None:
            raise AgentWorkspaceError("Graph revision not found.")
        document = _load(document_row["document_json"], {})
        _validate_schema_value(input_data, document.get("inputSchema", {}), "Graph input")
        if idempotency_key is not None:
            idempotency_key = _string(idempotency_key, "Graph idempotency key", maximum = 256)
            existing = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? AND graph_id = ? AND idempotency_key = ?",
                (project_id, graph_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return _run(existing)
        run_id = str(uuid.uuid4())
        current = now_ms()
        conn.execute(
            "INSERT INTO agent_graph_runs(id, project_id, graph_id, revision, input_json, status, retry_of_run_id, "
            "idempotency_key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
            (
                run_id,
                project_id,
                graph_id,
                selected_revision,
                _json(input_data, limit = _MAX_JSON_BYTES, label = "Graph run input"),
                retry_of_run_id,
                idempotency_key,
                current,
                current,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_graph_run(project_id: str, run_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graph_runs WHERE id = ? AND project_id = ?", (run_id, project_id)
        ).fetchone()
        return _run(row) if row else None
    finally:
        conn.close()


def list_graph_runs(
    project_id: str,
    graph_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    conn = _conn()
    try:
        if graph_id is None:
            rows = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, max(1, min(limit, 500))),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_graph_runs WHERE project_id = ? AND graph_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, graph_id, max(1, min(limit, 500))),
            ).fetchall()
        return [_run(row) for row in rows]
    finally:
        conn.close()


def claim_graph_run(run_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        current = now_ms()
        cursor = conn.execute(
            "UPDATE agent_graph_runs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (current, current, run_id),
        )
        conn.commit()
        if not cursor.rowcount:
            return None
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row else None
    finally:
        conn.close()


def update_graph_run(
    run_id: str,
    *,
    status: Optional[str] = None,
    output: Any = None,
    error: Optional[str] = None,
    current_node_id: Any = _UNSET,
) -> Optional[dict]:
    if status is not None and status not in _GRAPH_STATUSES:
        raise AgentWorkspaceError("Invalid graph run status.")
    if output is not None:
        _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph run output")
    assignments = ["updated_at = ?"]
    values: list[Any] = [now_ms()]
    if status is not None:
        assignments.append("status = ?")
        values.append(status)
        if status in {"cancelled", "completed", "failed", "interrupted"}:
            assignments.append("completed_at = ?")
            values.append(now_ms())
    if output is not None:
        assignments.append("output_json = ?")
        values.append(_json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph run output"))
    if error is not None:
        assignments.append("error = ?")
        values.append(str(error)[:8000])
    if current_node_id is not _UNSET:
        assignments.append("current_node_id = ?")
        values.append(current_node_id)
    values.append(run_id)
    conn = _conn()
    try:
        conn.execute(f"UPDATE agent_graph_runs SET {', '.join(assignments)} WHERE id = ?", values)
        conn.commit()
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        return _run(row) if row else None
    finally:
        conn.close()


def request_graph_pause(run_id: str) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["status"] == "queued":
            status = "paused"
        elif row["status"] == "running":
            status = "pausing"
        else:
            raise AgentWorkspaceError("Only queued or running graph runs can be paused.")
        conn.execute(
            "UPDATE agent_graph_runs SET status = ?, pause_requested = 1, updated_at = ? WHERE id = ?",
            (status, now_ms(), run_id),
        )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    finally:
        conn.close()


def resume_graph_run(run_id: str) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["status"] not in {"paused", "interrupted"}:
            raise AgentWorkspaceError("Only paused or interrupted graph runs can be resumed.")
        conn.execute(
            "UPDATE agent_graph_runs SET status = 'queued', pause_requested = 0, cancel_requested = 0, "
            "error = NULL, updated_at = ? WHERE id = ?",
            (now_ms(), run_id),
        )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    finally:
        conn.close()


def request_graph_cancel(run_id: str) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph run not found.")
        if row["status"] == "queued":
            status = "cancelled"
        elif row["status"] == "paused":
            status = "cancelled"
        elif row["status"] in {"running", "pausing"}:
            status = "cancelling"
        else:
            raise AgentWorkspaceError("Only active graph runs can be cancelled.")
        completed_at = now_ms() if status == "cancelled" else None
        if completed_at is None:
            conn.execute(
                "UPDATE agent_graph_runs SET status = ?, cancel_requested = 1, updated_at = ? WHERE id = ?",
                (status, now_ms(), run_id),
            )
        else:
            conn.execute(
                "UPDATE agent_graph_runs SET status = ?, cancel_requested = 1, updated_at = ?, completed_at = ? WHERE id = ?",
                (status, completed_at, completed_at, run_id),
            )
        conn.commit()
        return _run(
            conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
        )
    finally:
        conn.close()


def create_node_execution(run_id: str, node: dict, input_value: Any, attempt: int) -> dict:
    execution_id = str(uuid.uuid4())
    current = now_ms()
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO agent_graph_node_executions(id, run_id, node_id, node_type, attempt, status, input_json, created_at, started_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)",
            (
                execution_id,
                run_id,
                node["id"],
                node["type"],
                attempt,
                _json(input_value, limit = _MAX_JSON_BYTES, label = "Graph node input"),
                current,
                current,
            ),
        )
        conn.commit()
        return {
            "id": execution_id,
            "runId": run_id,
            "nodeId": node["id"],
            "nodeType": node["type"],
            "attempt": attempt,
            "status": "running",
        }
    finally:
        conn.close()


def finish_node_execution(
    execution_id: str,
    status: str,
    *,
    output: Any = None,
    error: Optional[str] = None,
) -> None:
    if status not in _NODE_STATUSES:
        raise AgentWorkspaceError("Invalid graph node status.")
    if output is not None:
        _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph node output")
    conn = _conn()
    try:
        conn.execute(
            "UPDATE agent_graph_node_executions SET status = ?, output_json = ?, error = ?, completed_at = ? WHERE id = ?",
            (
                status,
                _json(output, limit = _MAX_RUN_OUTPUT_BYTES, label = "Graph node output")
                if output is not None
                else None,
                str(error)[:8000] if error else None,
                now_ms(),
                execution_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_node_executions(project_id: str, run_id: str) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT e.* FROM agent_graph_node_executions e JOIN agent_graph_runs r ON r.id = e.run_id "
            "WHERE e.run_id = ? AND r.project_id = ? ORDER BY e.created_at, e.id",
            (run_id, project_id),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "runId": row["run_id"],
                "nodeId": row["node_id"],
                "nodeType": row["node_type"],
                "attempt": row["attempt"],
                "status": row["status"],
                "input": _load(row["input_json"], None),
                "output": _load(row["output_json"], None),
                "error": row["error"],
                "createdAt": row["created_at"],
                "startedAt": row["started_at"],
                "completedAt": row["completed_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def append_graph_event(
    run_id: str,
    event_type: str,
    *,
    node_id: Optional[str] = None,
    payload: Optional[dict] = None,
) -> dict:
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        sequence_row = conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM agent_graph_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        sequence = int(sequence_row["next"])
        event_id = str(uuid.uuid4())
        current = now_ms()
        conn.execute(
            "INSERT INTO agent_graph_events(id, run_id, sequence, event_type, node_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                run_id,
                sequence,
                _string(event_type, "Graph event type", maximum = 128),
                node_id,
                _json(payload or {}, limit = 64 * 1024, label = "Graph event"),
                current,
            ),
        )
        conn.commit()
        return {
            "id": event_id,
            "runId": run_id,
            "sequence": sequence,
            "type": event_type,
            "nodeId": node_id,
            "payload": payload or {},
            "createdAt": current,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_graph_events(
    project_id: str,
    run_id: str,
    after: int = 0,
    limit: int = 500,
) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT e.* FROM agent_graph_events e JOIN agent_graph_runs r ON r.id = e.run_id "
            "WHERE e.run_id = ? AND r.project_id = ? AND e.sequence > ? ORDER BY e.sequence LIMIT ?",
            (run_id, project_id, max(0, after), max(1, min(limit, 1000))),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "runId": row["run_id"],
                "sequence": row["sequence"],
                "type": row["event_type"],
                "nodeId": row["node_id"],
                "payload": _load(row["payload_json"], {}),
                "createdAt": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_or_create_approval(project_id: str, run_id: str, node: dict) -> dict:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graph_approvals WHERE run_id = ? AND node_id = ?",
            (run_id, node["id"]),
        ).fetchone()
        if row is None:
            approval_id = str(uuid.uuid4())
            current = now_ms()
            config = node["config"]
            conn.execute(
                "INSERT INTO agent_graph_approvals(id, project_id, run_id, node_id, title, description, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    approval_id,
                    project_id,
                    run_id,
                    node["id"],
                    config["title"],
                    config["description"],
                    current,
                    current,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_graph_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        return _approval(row)
    finally:
        conn.close()


def _approval(row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "runId": row["run_id"],
        "nodeId": row["node_id"],
        "title": row["title"],
        "description": row["description"],
        "status": row["status"],
        "decision": row["decision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def get_graph_approval(project_id: str, run_id: str, approval_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM agent_graph_approvals WHERE id = ? AND project_id = ? AND run_id = ?",
            (approval_id, project_id, run_id),
        ).fetchone()
        return _approval(row) if row else None
    finally:
        conn.close()


def list_graph_approvals(project_id: str, run_id: str) -> list[dict]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT a.* FROM agent_graph_approvals a JOIN agent_graph_runs r ON r.id = a.run_id "
            "WHERE a.project_id = ? AND a.run_id = ? ORDER BY a.created_at, a.id",
            (project_id, run_id),
        ).fetchall()
        return [_approval(row) for row in rows]
    finally:
        conn.close()


def decide_graph_approval(project_id: str, run_id: str, approval_id: str, decision: str) -> dict:
    if decision not in {"approved", "rejected"}:
        raise AgentWorkspaceError("Approval decision is invalid.")
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_graph_approvals WHERE id = ? AND project_id = ? AND run_id = ?",
            (approval_id, project_id, run_id),
        ).fetchone()
        if row is None:
            raise AgentWorkspaceError("Graph approval not found.")
        if row["status"] != "pending":
            raise AgentWorkspaceError("Graph approval has already been decided.")
        current = now_ms()
        conn.execute(
            "UPDATE agent_graph_approvals SET status = ?, decision = ?, updated_at = ? WHERE id = ?",
            (decision, decision, current, approval_id),
        )
        conn.commit()
        result = _approval(
            conn.execute(
                "SELECT * FROM agent_graph_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        )
    finally:
        conn.close()
    append_graph_event(
        run_id,
        "approval.decided",
        node_id = result["nodeId"],
        payload = {"approvalId": approval_id, "decision": decision},
    )
    return result


def recover_graph_runs() -> int:
    """Fence in-flight graph records after a process restart."""
    conn = _conn()
    try:
        current = now_ms()
        cursor = conn.execute(
            "UPDATE agent_graph_runs SET status = 'interrupted', error = COALESCE(error, 'Studio restarted while the graph was active.'), "
            "updated_at = ?, completed_at = ? WHERE status IN ('running', 'pausing', 'cancelling')",
            (current, current),
        )
        conn.execute(
            "UPDATE agent_graph_node_executions SET status = 'interrupted', error = COALESCE(error, 'Studio restarted while the graph was active.'), completed_at = ? WHERE status = 'running'",
            (current,),
        )
        conn.commit()
        return int(cursor.rowcount)
    finally:
        conn.close()


def _graph_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def _render(value: Any, context: dict) -> Any:
    if isinstance(value, str):
        try:
            fields = {field for _, field, _, _ in Formatter().parse(value) if field}
            if not fields:
                return value
            replacements = {
                field: json.dumps(_graph_path(context, field), ensure_ascii = False)
                for field in fields
            }
            return value.format(**replacements)
        except (KeyError, ValueError, IndexError) as exc:
            raise AgentWorkspaceError("Graph template references an invalid context path.") from exc
    if isinstance(value, dict):
        return {key: _render(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    return value


def _condition(value: Any, config: dict) -> bool:
    operator = config["operator"]
    if operator == "truthy":
        return bool(value)
    if operator == "falsy":
        return not bool(value)
    if operator == "exists":
        return value is not None
    if operator == "equals":
        return value == config.get("value")
    return value != config.get("value")


def _validate_schema_value(value: Any, schema: dict, label: str) -> None:
    """Validate the bounded JSON-schema subset used by graph inputs and outputs."""
    if not isinstance(schema, dict):
        raise AgentWorkspaceError(f"{label} schema is invalid.")
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected is not None and expected not in type_matches:
        raise AgentWorkspaceError(f"{label} schema type is invalid.")
    if expected is not None and not type_matches[expected]:
        raise AgentWorkspaceError(f"{label} does not match the graph schema.")
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise AgentWorkspaceError(f"{label} schema required fields are invalid.")
        missing = [item for item in required if item not in value]
        if missing:
            raise AgentWorkspaceError(f"{label} is missing required fields: {', '.join(missing)}.")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise AgentWorkspaceError(f"{label} schema properties are invalid.")
        if schema.get("additionalProperties") is False:
            unknown = [key for key in value if key not in properties]
            if unknown:
                raise AgentWorkspaceError(
                    f"{label} contains unsupported fields: {', '.join(unknown)}."
                )
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema_value(value[key], child_schema, f"{label}.{key}")
    elif isinstance(value, list) and schema.get("items") is not None:
        if not isinstance(schema["items"], dict):
            raise AgentWorkspaceError(f"{label} schema items are invalid.")
        for index, item in enumerate(value):
            _validate_schema_value(item, schema["items"], f"{label}[{index}]")


def _validate_schema_definition(schema: dict, label: str) -> None:
    if not isinstance(schema, dict):
        raise AgentWorkspaceError(f"{label} schema is invalid.")
    expected = schema.get("type")
    if expected is not None and expected not in {
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
    }:
        raise AgentWorkspaceError(f"{label} schema type is invalid.")
    if expected == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise AgentWorkspaceError(f"{label} schema is invalid.")
        if any(not isinstance(item, str) or item not in properties for item in required):
            raise AgentWorkspaceError(f"{label} schema required fields are invalid.")
        for key, child in properties.items():
            if not isinstance(key, str):
                raise AgentWorkspaceError(f"{label} schema property names are invalid.")
            _validate_schema_definition(child, f"{label}.{key}")
    if expected == "array" and schema.get("items") is not None:
        _validate_schema_definition(schema["items"], f"{label} items")


class GraphLoopAdapter:
    """Adapter boundary for the existing durable background-agent runtime."""

    def run(
        self,
        project_id: str,
        instruction: str,
        runtime: Optional[dict],
        cancel_event: threading.Event,
    ) -> dict:
        from .background import manager as background_manager
        task = background_manager.enqueue_agent(
            project_id,
            instruction,
            runtime_selection = runtime,
            start = True,
        )
        while True:
            if cancel_event.is_set():
                background_manager.cancel(task["id"])
            current = background_manager_task(task["id"])
            if current is None:
                raise AgentWorkspaceError("Graph loop task disappeared before completion.")
            if current["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                if current["status"] != "completed":
                    raise AgentWorkspaceError(current.get("error") or "Graph loop failed.")
                return current.get("result") or {}
            time.sleep(0.025)


def background_manager_task(task_id: str) -> Optional[dict]:
    from .state import get_background_task
    return get_background_task(task_id)


class GraphRunManager:
    """Durable graph coordinator that delegates node work to existing runtimes."""

    def __init__(
        self,
        max_workers: int = 2,
        loop_adapter: Optional[GraphLoopAdapter] = None,
    ):
        self._executor = ThreadPoolExecutor(
            max_workers = max_workers, thread_name_prefix = "studio-graph-run"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._deleting_projects: set[str] = set()
        self.loop_adapter = loop_adapter or GraphLoopAdapter()

    def begin_project_deletion(self, project_id: str) -> None:
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is already in progress.")
            self._deleting_projects.add(project_id)

    def finish_project_deletion(self, project_id: str) -> None:
        with self._lock:
            self._deleting_projects.discard(project_id)

    def enqueue(
        self,
        project_id: str,
        graph_id: str,
        input_data: dict,
        *,
        revision: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        start: bool = True,
    ) -> dict:
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is in progress.")
            run = create_graph_run(
                project_id, graph_id, input_data, revision = revision, idempotency_key = idempotency_key
            )
        if start and run["status"] == "queued":
            return self.start(run["id"])
        return run

    def start(self, run_id: str) -> dict:
        with self._lock:
            claimed = claim_graph_run(run_id)
            if claimed is None:
                run = self._get_any(run_id)
                if run is None:
                    raise AgentWorkspaceError("Graph run not found.")
                if run["status"] != "running":
                    raise AgentWorkspaceError("Only queued graph runs can be started.")
                return run
            cancel_event = threading.Event()
            self._cancellations[run_id] = cancel_event
            try:
                future = self._executor.submit(self._run, run_id, cancel_event)
            except Exception as exc:
                self._cancellations.pop(run_id, None)
                update_graph_run(
                    run_id,
                    status = "failed",
                    error = "The graph coordinator could not start this run.",
                )
                raise AgentWorkspaceError(
                    "The graph coordinator could not start this run."
                ) from exc
            self._futures[run_id] = future
            future.add_done_callback(lambda _future: self._forget(run_id))
            return claimed

    def _get_any(self, run_id: str) -> Optional[dict]:
        conn = _conn()
        try:
            row = conn.execute("SELECT * FROM agent_graph_runs WHERE id = ?", (run_id,)).fetchone()
            return _run(row) if row else None
        finally:
            conn.close()

    def pause(self, run_id: str) -> dict:
        run = request_graph_pause(run_id)
        event = self._cancellations.get(run_id)
        if event:
            event.set()
        return run

    def resume(self, run_id: str) -> dict:
        run = resume_graph_run(run_id)
        return self.start(run_id)

    def cancel(self, run_id: str) -> dict:
        run = request_graph_cancel(run_id)
        event = self._cancellations.get(run_id)
        if event:
            event.set()
        return run

    def retry(
        self,
        project_id: str,
        run_id: str,
        *,
        start: bool = True,
    ) -> dict:
        previous = get_graph_run(project_id, run_id)
        if previous is None:
            raise AgentWorkspaceError("Graph run not found.")
        if previous["status"] not in {"failed", "cancelled", "interrupted"}:
            raise AgentWorkspaceError("Only stopped graph runs can be retried.")
        with self._lock:
            if project_id in self._deleting_projects:
                raise AgentWorkspaceError("Project deletion is in progress.")
            run = create_graph_run(
                project_id,
                previous["graphId"],
                previous["input"],
                revision = previous["revision"],
                retry_of_run_id = run_id,
            )
        return self.start(run["id"]) if start else run

    def _forget(self, run_id: str) -> None:
        with self._lock:
            self._futures.pop(run_id, None)
            self._cancellations.pop(run_id, None)

    def prepare_for_app_exit(self, timeout_seconds: float = 10) -> None:
        with self._lock:
            events = list(self._cancellations.values())
            futures = list(self._futures.values())
        for event in events:
            event.set()
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                future.result(timeout = remaining)
            except Exception:
                pass
        recover_graph_runs()

    def cancel_project_runs_and_wait(
        self,
        project_id: str,
        timeout_seconds: float = 30,
    ) -> list[dict]:
        active = [
            run
            for run in list_graph_runs(project_id, limit = 500)
            if run["status"] in {"queued", "running", "pausing", "cancelling"}
        ]
        for run in active:
            try:
                self.cancel(run["id"])
            except AgentWorkspaceError:
                pass
        with self._lock:
            futures = [self._futures[run["id"]] for run in active if run["id"] in self._futures]
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentWorkspaceError("Timed out while stopping project graph runs.")
            try:
                future.result(timeout = remaining)
            except Exception as exc:
                raise AgentWorkspaceError("Failed while stopping a project graph run.") from exc
        remaining_runs = [
            run
            for run in list_graph_runs(project_id, limit = 500)
            if run["status"] in {"queued", "running", "pausing", "cancelling"}
        ]
        if remaining_runs:
            raise AgentWorkspaceError("Project still has active graph runs.")
        return [
            get_graph_run(project_id, run["id"])
            for run in active
            if get_graph_run(project_id, run["id"])
        ]

    def _run(self, run_id: str, cancel_event: threading.Event) -> None:
        run = self._get_any(run_id)
        if run is None:
            return
        graph = get_graph_revision(run["projectId"], run["graphId"], run["revision"])
        budget_expired = threading.Event()
        timer = threading.Timer(
            int((graph or {}).get("limits", {}).get("maxRunSeconds", 3600)),
            lambda: (budget_expired.set(), cancel_event.set()),
        )
        timer.daemon = True
        timer.start()
        try:
            self._run_impl(run_id, cancel_event, budget_expired)
        finally:
            timer.cancel()

    def _run_impl(
        self, run_id: str, cancel_event: threading.Event, budget_expired: threading.Event
    ) -> None:
        run = self._get_any(run_id)
        if run is None:
            return
        try:
            graph = get_graph_revision(run["projectId"], run["graphId"], run["revision"])
            if graph is None:
                raise AgentWorkspaceError("Pinned graph revision is unavailable.")
            nodes = {node["id"]: node for node in graph["nodes"]}
            edges = graph["edges"]
            executions = list_node_executions(run["projectId"], run_id)
            context = {"input": run["input"], "nodes": {}, "previous": None}
            for execution in executions:
                if execution["status"] == "completed":
                    context["nodes"][execution["nodeId"]] = execution["output"]
                    context["previous"] = execution["output"]
            current_node_id = run.get("currentNodeId")
            if current_node_id is None:
                completed = [item for item in executions if item["status"] == "completed"]
                if completed:
                    current_node_id = self._next_node(
                        completed[-1]["nodeId"], completed[-1]["output"], nodes, edges
                    )
                else:
                    current_node_id = next(
                        node["id"] for node in graph["nodes"] if node["type"] == "input"
                    )
            node_count = len([item for item in executions if item["status"] == "completed"])
            while current_node_id is not None:
                run = self._get_any(run_id) or run
                if budget_expired.is_set():
                    raise AgentWorkspaceError("Graph run budget exhausted.")
                if run["cancelRequested"] or (cancel_event.is_set() and not run["pauseRequested"]):
                    update_graph_run(run_id, status = "cancelled", error = "Graph run cancelled.")
                    append_graph_event(run_id, "run.cancelled")
                    return
                if (
                    run["pauseRequested"]
                    or run["status"] == "pausing"
                    or (cancel_event.is_set() and run["status"] == "paused")
                ):
                    update_graph_run(run_id, status = "paused", current_node_id = current_node_id)
                    append_graph_event(run_id, "run.paused", node_id = current_node_id)
                    return
                if node_count >= graph["limits"]["maxNodes"]:
                    raise AgentWorkspaceError("Graph node budget exhausted.")
                node = nodes[current_node_id]
                update_graph_run(run_id, current_node_id = current_node_id)
                append_graph_event(
                    run_id, "node.started", node_id = current_node_id, payload = {"type": node["type"]}
                )
                prior_attempts = [item for item in executions if item["nodeId"] == current_node_id]
                execution = create_node_execution(
                    run_id, node, context["previous"], len(prior_attempts) + 1
                )
                node_timed_out = threading.Event()
                node_timer = threading.Timer(
                    int(node["config"].get("timeoutSeconds", _MAX_NODE_SECONDS)),
                    lambda: (node_timed_out.set(), cancel_event.set()),
                )
                node_timer.daemon = True
                node_timer.start()
                try:
                    output = self._execute_node(run, graph, node, context, cancel_event)
                    _json(
                        output,
                        limit = graph["limits"]["maxOutputBytes"],
                        label = "Graph node output",
                    )
                    if node_timed_out.is_set():
                        raise AgentWorkspaceError("Graph node timeout exceeded.")
                    if cancel_event.is_set() and ((self._get_any(run_id) or run)["pauseRequested"]):
                        finish_node_execution(execution["id"], "paused", output = output)
                        update_graph_run(run_id, status = "paused", current_node_id = current_node_id)
                        append_graph_event(run_id, "node.paused", node_id = current_node_id)
                        return
                    finish_node_execution(execution["id"], "completed", output = output)
                    executions.append(
                        {"nodeId": current_node_id, "status": "completed", "output": output}
                    )
                    context["nodes"][current_node_id] = output
                    context["previous"] = output
                    node_count += 1
                    append_graph_event(run_id, "node.completed", node_id = current_node_id)
                    current_node_id = self._next_node(current_node_id, output, nodes, edges)
                    update_graph_run(run_id, current_node_id = current_node_id)
                except Exception as exc:
                    current = self._get_any(run_id) or run
                    if budget_expired.is_set():
                        finish_node_execution(
                            execution["id"], "failed", error = "Graph run budget exhausted."
                        )
                        raise AgentWorkspaceError("Graph run budget exhausted.") from exc
                    if current["pauseRequested"]:
                        finish_node_execution(execution["id"], "paused", error = str(exc))
                        update_graph_run(run_id, status = "paused", current_node_id = current_node_id)
                        append_graph_event(run_id, "node.paused", node_id = current_node_id)
                        return
                    finish_node_execution(
                        execution["id"],
                        "cancelled" if current["cancelRequested"] else "failed",
                        error = str(exc),
                    )
                    status = "cancelled" if current["cancelRequested"] else "failed"
                    update_graph_run(run_id, status = status, error = str(exc))
                    append_graph_event(
                        run_id,
                        "run." + status,
                        node_id = current_node_id,
                        payload = {"error": str(exc)[:1000]},
                    )
                    return
                finally:
                    node_timer.cancel()
            final_output = context["previous"] or {}
            _json(
                final_output,
                limit = graph["limits"]["maxOutputBytes"],
                label = "Graph output",
            )
            _validate_schema_value(final_output, graph.get("outputSchema", {}), "Graph output")
            update_graph_run(run_id, status = "completed", output = final_output)
            append_graph_event(run_id, "run.completed")
        except Exception as exc:
            current = self._get_any(run_id)
            if current and current["status"] not in {"cancelled", "completed"}:
                update_graph_run(run_id, status = "failed", error = str(exc))
                append_graph_event(run_id, "run.failed", payload = {"error": str(exc)[:1000]})

    def _execute_node(
        self, run: dict, graph: dict, node: dict, context: dict, cancel_event: threading.Event
    ) -> Any:
        node_type = node["type"]
        config = node["config"]
        if node_type == "input":
            return context["input"]
        if node_type in {"loop", "model"}:
            template = config["instruction"] if node_type == "loop" else config["prompt"]
            instruction = _render(template, context)
            return self.loop_adapter.run(
                run["projectId"], instruction, config.get("runtime"), cancel_event
            )
        if node_type == "tool":
            allowed = graph["permissions"].get("allowedToolServerIds", [])
            if config["serverId"] not in allowed:
                raise AgentWorkspaceError("Graph tool is not permitted by this graph revision.")
            server = mcp_servers_db.get_server(config["serverId"])
            if server is None or not server.get("is_enabled"):
                raise AgentWorkspaceError("Graph tool server is unavailable.")
            return call_tool_sync(
                server["url"],
                parse_server_headers(server),
                config["toolName"],
                _render(config["arguments"], context),
                timeout = config["timeoutSeconds"],
                use_oauth = bool(server.get("use_oauth")),
                cancel_event = cancel_event,
                scope = f"graph:{run['id']}",
            )
        if node_type == "condition":
            return _condition(_graph_path(context, config["path"]), config)
        if node_type == "approval":
            approval = get_or_create_approval(run["projectId"], run["id"], node)
            append_graph_event(
                run["id"],
                "approval.required",
                node_id = node["id"],
                payload = {"approvalId": approval["id"]},
            )
            while approval["status"] == "pending":
                current = self._get_any(run["id"]) or run
                if current["cancelRequested"]:
                    raise AgentWorkspaceError("Graph approval was cancelled.")
                if current["pauseRequested"] or cancel_event.is_set():
                    raise AgentWorkspaceError("Graph approval was paused.")
                time.sleep(0.05)
                approval = (
                    get_graph_approval(run["projectId"], run["id"], approval["id"]) or approval
                )
            if approval["status"] == "rejected":
                raise AgentWorkspaceError("Graph approval was rejected.")
            return {"approvalId": approval["id"], "decision": "approved"}
        if node_type == "output":
            return (
                _graph_path(context, config["path"]) if config.get("path") else context["previous"]
            )
        raise AgentWorkspaceError("Unsupported graph node type.")

    @staticmethod
    def _next_node(
        node_id: str, output: Any, nodes: dict[str, dict], edges: list[dict]
    ) -> Optional[str]:
        outgoing = [edge for edge in edges if edge["from"] == node_id]
        if not outgoing:
            return None
        if nodes[node_id]["type"] != "condition":
            return outgoing[0]["to"]
        if len(outgoing) == 1 and outgoing[0].get("when") in {None, "default"}:
            return outgoing[0]["to"]
        wanted = "true" if bool(output) else "false"
        for edge in outgoing:
            if edge.get("when") == wanted:
                return edge["to"]
        for edge in outgoing:
            if edge.get("when") == "default":
                return edge["to"]
        return None


manager = GraphRunManager()


__all__ = [
    "GraphLoopAdapter",
    "GraphRunManager",
    "append_graph_event",
    "create_graph",
    "create_graph_run",
    "delete_graph",
    "decide_graph_approval",
    "get_graph",
    "get_graph_approval",
    "get_graph_revision",
    "get_graph_run",
    "list_graph_events",
    "list_graph_approvals",
    "list_graph_runs",
    "list_graphs",
    "list_node_executions",
    "manager",
    "recover_graph_runs",
    "update_graph",
    "validate_graph_spec",
]
