# Sloth Graphs

Sloth Graphs are durable, project-scoped workflows for composing existing Sloth Loop style agent work with model, tool, condition, approval, and output nodes.

The graph coordinator is not a second agent runtime. Loop and model nodes submit work through `core.agent_workspace.background.BackgroundTaskManager`, which remains the single provider-neutral execution boundary. Tool nodes use the existing MCP client and an explicit graph permission allow-list.

## Contract

Each graph has an immutable, numbered revision. A revision contains:

- typed nodes and directed edges;
- input and output JSON schemas;
- tool-server permissions;
- bounded node count, run time, and output size;
- project ownership and creation timestamps.

The validator rejects duplicate IDs, dangling edges, self edges, cycles, unreachable nodes, unsupported joins, invalid node configuration, and graphs that do not start at one input node and terminate at an output node. Condition nodes may select a true or false edge.

Graph runs pin the selected revision at creation. Durable state includes the run, every node execution attempt, append-only events, and approval decisions. Run actions are project-scoped and support queued execution, inspection, pause, resume, cancellation, retry lineage, idempotency keys, and restart-to-interrupted recovery.

Templates may reference `input`, `previous`, and `nodes.<id>` in loop instructions, model prompts, and tool arguments. Approval nodes remain pending until an authenticated project user decides them. Tool server IDs must appear in `permissions.allowedToolServerIds`.

## API and Studio

The authenticated routes are under `/api/agent-workspace/projects/{project_id}`:

- `POST /graphs`, `GET /graphs`, `GET /graphs/{graph_id}`, `PUT /graphs/{graph_id}`, and `DELETE /graphs/{graph_id}`. Deletion is refused while a run is active and removes stopped run history with the graph;
- `POST /graphs/{graph_id}/runs` and `GET /graphs/{graph_id}/runs`;
- `GET /graph-runs/{run_id}`, `GET /graph-runs/{run_id}/events`;
- `POST /graph-runs/{run_id}/pause`, `resume`, `cancel`, and `retry`;
- `POST /graph-runs/{run_id}/approvals/{approval_id}`.

The Agent Workspace panel includes a JSON revision editor, run input, run controls, node execution inspection, event history, and approval actions. The backend remains authoritative for validation, revision pinning, project scope, permissions, and state transitions.

## Verification boundary

Focused validation covers graph persistence, revision pinning, DAG rejection, schema enforcement, sequential execution through the existing adapter boundary, pause and resume, approvals, restart recovery, project scoping, idempotent starts, and the existing durable agent suites. Real provider, MCP, packaged desktop, and physical-platform runs remain separate release gates.
