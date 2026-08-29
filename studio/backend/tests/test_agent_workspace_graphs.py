# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.graphs import (
    GraphLoopAdapter,
    GraphRunManager,
    create_graph,
    create_graph_run,
    delete_graph,
    decide_graph_approval,
    get_graph_approval,
    get_graph_run,
    list_graph_events,
    list_node_executions,
    recover_graph_runs,
    update_graph,
    validate_graph_spec,
)
from storage import studio_db
from routes import agent_workspace as agent_workspace_routes


def _folder_project(root, project_id: str = "project") -> dict:
    metadata = root.stat()
    return studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Graph goal",
            "goalStatus": "active",
            "goalUpdatedAt": 1,
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _spec(
    *nodes,
    edges = None,
    name = "Graph",
):
    return {
        "name": name,
        "nodes": list(nodes),
        "edges": edges or [],
    }


def _node(
    node_id,
    node_type,
    config = None,
):
    return {"id": node_id, "type": node_type, "config": config or {}}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_workspace_routes.router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def _wait(run_id, timeout = 5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = get_graph_run("project", run_id)
        if run and run["status"] in {"paused", "cancelled", "completed", "failed", "interrupted"}:
            return run
        time.sleep(0.01)
    raise AssertionError("graph run did not stop")


class _EchoAdapter(GraphLoopAdapter):
    def __init__(self):
        self.instructions = []

    def run(self, project_id, instruction, runtime, cancel_event):
        self.instructions.append((project_id, instruction, runtime))
        return {"output": instruction}


def test_graph_validation_rejects_dangling_cycles_duplicates_and_unreachable():
    root = _node("input", "input")
    output = _node("output", "output")
    with pytest.raises(AgentWorkspaceError, match = "existing nodes"):
        validate_graph_spec(_spec(root, output, edges = [{"from": "input", "to": "missing"}]))
    with pytest.raises(AgentWorkspaceError, match = "root input"):
        validate_graph_spec(
            _spec(
                root,
                output,
                edges = [
                    {"from": "input", "to": "output"},
                    {"from": "output", "to": "input"},
                ],
            )
        )
    with pytest.raises(AgentWorkspaceError, match = "unique"):
        validate_graph_spec(_spec(root, root, output, edges = [{"from": "input", "to": "output"}]))
    with pytest.raises(AgentWorkspaceError, match = "unreachable"):
        validate_graph_spec(
            _spec(root, output, _node("dead", "output"), edges = [{"from": "input", "to": "output"}])
        )
    with pytest.raises(AgentWorkspaceError, match = "single condition edge"):
        validate_graph_spec(
            _spec(
                root,
                _node("check", "condition", {"path": "input.ok"}),
                output,
                edges = [
                    {"from": "input", "to": "check"},
                    {"from": "check", "to": "output", "when": "true"},
                ],
            )
        )


def test_condition_node_selects_a_validated_branch(tmp_path):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("check", "condition", {"path": "input.ok"}),
                _node("yes", "output", {"path": "input"}),
                _node("no", "output", {"path": "input"}),
                edges = [
                    {"from": "input", "to": "check"},
                    {"from": "check", "to": "yes", "when": "true"},
                    {"from": "check", "to": "no", "when": "false"},
                ],
            ),
        )
        true_run = manager.enqueue("project", graph["id"], {"ok": True})
        false_run = manager.enqueue("project", graph["id"], {"ok": False})
        assert _wait(true_run["id"])["output"] == {"ok": True}
        assert _wait(false_run["id"])["output"] == {"ok": False}
        assert [item["nodeId"] for item in list_node_executions("project", true_run["id"])] == [
            "input",
            "check",
            "yes",
        ]
        assert [item["nodeId"] for item in list_node_executions("project", false_run["id"])] == [
            "input",
            "check",
            "no",
        ]
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_enforces_revision_output_budget(tmp_path):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            {
                **_spec(
                    _node("input", "input"),
                    _node("output", "output", {"path": "input"}),
                    edges = [{"from": "input", "to": "output"}],
                ),
                "limits": {"maxNodes": 2, "maxRunSeconds": 60, "maxOutputBytes": 1024},
            },
        )
        run = manager.enqueue("project", graph["id"], {"value": "x" * 2000})
        finished = _wait(run["id"])
        assert finished["status"] == "failed"
        assert "too large" in (finished["error"] or "")
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_delete_preserves_active_runs_until_stopped(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = create_graph_run("project", graph["id"], {})
    with pytest.raises(AgentWorkspaceError, match = "active graph runs"):
        delete_graph("project", graph["id"])
    manager = GraphRunManager(max_workers = 1)
    try:
        manager.start(run["id"])
        assert _wait(run["id"])["status"] == "completed"
        delete_graph("project", graph["id"])
        assert get_graph_run("project", run["id"]) is None
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_revisions_are_immutable_and_runs_pin_revision(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    run = GraphRunManager(max_workers = 1).enqueue(
        "project", graph["id"], {"value": "before"}, start = False
    )
    updated = update_graph(
        "project",
        graph["id"],
        _spec(
            _node("input", "input"),
            _node("output", "output", {"path": "input.value"}),
            edges = [{"from": "input", "to": "output"}],
            name = "Graph v2",
        ),
        expected_revision = 1,
    )
    assert updated["currentRevision"] == 2
    assert run["revision"] == 1


def test_graph_run_enforces_revision_input_schema(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        {
            **_spec(
                _node("input", "input"),
                _node("output", "output"),
                edges = [{"from": "input", "to": "output"}],
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    )
    with pytest.raises(AgentWorkspaceError, match = "missing required"):
        create_graph_run("project", graph["id"], {})
    run = create_graph_run("project", graph["id"], {"value": "ok"})
    assert run["revision"] == 1


def test_sequential_graph_uses_one_existing_loop_adapter_and_records_events(tmp_path):
    _folder_project(tmp_path)
    adapter = _EchoAdapter()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("loop", "loop", {"instruction": "echo {input}"}),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "loop"},
                    {"from": "loop", "to": "output"},
                ],
            ),
        )
        run = manager.enqueue("project", graph["id"], {"value": "x"})
        finished = _wait(run["id"])
        assert finished["status"] == "completed"
        assert finished["revision"] == 1
        assert finished["output"] == {"output": 'echo {"value": "x"}'}
        assert adapter.instructions == [("project", 'echo {"value": "x"}', None)]
        executions = list_node_executions("project", run["id"])
        assert [item["status"] for item in executions] == ["completed"] * 3
        events = list_graph_events("project", run["id"])
        assert [event["type"] for event in events].count("node.completed") == 3
        assert events[-1]["type"] == "run.completed"
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_pause_resume_retries_current_node(tmp_path):
    _folder_project(tmp_path)
    entered = threading.Event()

    class _PausingAdapter(GraphLoopAdapter):
        def __init__(self):
            self.calls = 0

        def run(self, project_id, instruction, runtime, cancel_event):
            self.calls += 1
            entered.set()
            if self.calls == 1:
                cancel_event.wait(timeout = 2)
            return {"output": f"call-{self.calls}"}

    adapter = _PausingAdapter()
    manager = GraphRunManager(max_workers = 1, loop_adapter = adapter)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("loop", "loop", {"instruction": "run"}),
                _node("output", "output"),
                edges = [{"from": "input", "to": "loop"}, {"from": "loop", "to": "output"}],
            ),
        )
        run = manager.enqueue("project", graph["id"], {})
        assert entered.wait(timeout = 2)
        paused = manager.pause(run["id"])
        assert paused["status"] in {"pausing", "paused"}
        stopped = _wait(run["id"])
        assert stopped["status"] == "paused"
        resumed = manager.resume(run["id"])
        assert resumed["status"] == "running"
        assert _wait(run["id"])["status"] == "completed"
        assert adapter.calls == 2
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_approval_blocks_until_decision(tmp_path):
    _folder_project(tmp_path)
    manager = GraphRunManager(max_workers = 1)
    try:
        graph = create_graph(
            "project",
            _spec(
                _node("input", "input"),
                _node("approval", "approval", {"title": "Ship it"}),
                _node("output", "output"),
                edges = [
                    {"from": "input", "to": "approval"},
                    {"from": "approval", "to": "output"},
                ],
            ),
        )
        run = manager.enqueue("project", graph["id"], {})
        deadline = time.monotonic() + 3
        approval = None
        while time.monotonic() < deadline:
            approval_events = [
                event
                for event in list_graph_events("project", run["id"])
                if event["type"] == "approval.required"
            ]
            if approval_events:
                approval = get_graph_approval(
                    "project", run["id"], approval_events[0]["payload"]["approvalId"]
                )
                break
            time.sleep(0.01)
        assert approval and approval["status"] == "pending"
        decided = decide_graph_approval("project", run["id"], approval["id"], "approved")
        assert decided["status"] == "approved"
        assert _wait(run["id"])["status"] == "completed"
        assert any(
            event["type"] == "approval.decided" for event in list_graph_events("project", run["id"])
        )
    finally:
        manager._executor.shutdown(wait = True)


def test_graph_recovery_marks_active_runs_interrupted(tmp_path):
    _folder_project(tmp_path)
    graph = create_graph(
        "project",
        _spec(
            _node("input", "input"),
            _node("output", "output"),
            edges = [{"from": "input", "to": "output"}],
        ),
    )
    manager = GraphRunManager(max_workers = 1)
    run = manager.enqueue("project", graph["id"], {}, start = False)
    from core.agent_workspace.graphs import claim_graph_run

    claim_graph_run(run["id"])
    manager._executor.shutdown(wait = True)
    assert recover_graph_runs() == 1
    assert get_graph_run("project", run["id"])["status"] == "interrupted"


def test_graph_api_is_project_scoped_and_pins_revision(tmp_path):
    _folder_project(tmp_path)
    client = _client()
    payload = _spec(
        _node("input", "input"),
        _node("output", "output"),
        edges = [{"from": "input", "to": "output"}],
    )
    response = client.post("/api/agent-workspace/projects/project/graphs", json = payload)
    assert response.status_code == 200
    graph = response.json()
    assert graph["currentRevision"] == 1

    run_response = client.post(
        f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
        json = {"input": {"value": "ok"}, "idempotencyKey": "request-1"},
    )
    assert run_response.status_code == 200
    run = run_response.json()
    assert run["revision"] == 1
    assert (
        client.post(
            f"/api/agent-workspace/projects/project/graphs/{graph['id']}/runs",
            json = {"input": {"value": "different"}, "idempotencyKey": "request-1"},
        ).json()["id"]
        == run["id"]
    )
    second_graph = create_graph("project", {**payload, "name": "Graph 2"})
    second_run = create_graph_run(
        "project",
        second_graph["id"],
        {"value": "other"},
        idempotency_key = "request-1",
    )
    assert second_run["id"] != run["id"]
    assert (
        client.get(f"/api/agent-workspace/projects/other/graphs/{graph['id']}").status_code == 404
    )
    assert _wait(run["id"])["status"] == "completed"
    assert client.delete(f"/api/agent-workspace/projects/project/graphs/{graph['id']}").json() == {
        "deleted": True
    }
    assert (
        client.get(f"/api/agent-workspace/projects/project/graphs/{graph['id']}").status_code == 404
    )
