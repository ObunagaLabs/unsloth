// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { CheckCircle2, Loader2, Pause, Play, Plus, RefreshCw, RotateCcw, Square, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  type AgentGraphApproval,
  type AgentGraphDocument,
  type AgentGraphEvent,
  type AgentGraphNodeExecution,
  type AgentGraphRevision,
  type AgentGraphRun,
  type AgentGraphSummary,
  cancelAgentGraphRun,
  createAgentGraph,
  deleteAgentGraph,
  decideAgentGraphApproval,
  getAgentGraph,
  getAgentGraphRun,
  listAgentGraphEvents,
  listAgentGraphRuns,
  listAgentGraphs,
  pauseAgentGraphRun,
  retryAgentGraphRun,
  resumeAgentGraphRun,
  startAgentGraphRun,
  updateAgentGraph,
} from "../api/agent-workspace-api";
import { safeAgentWorkspaceError } from "./agent-workspace-state";

const SAMPLE_GRAPH: AgentGraphDocument = {
  name: "New graph",
  description: "",
  inputSchema: { type: "object" },
  outputSchema: { type: "object" },
  nodes: [
    { id: "input", type: "input", config: { name: "input" } },
    { id: "loop", type: "loop", config: { instruction: "Work on {input}" } },
    { id: "output", type: "output", config: { name: "output" } },
  ],
  edges: [
    { from: "input", to: "loop" },
    { from: "loop", to: "output" },
  ],
  permissions: { allowedToolServerIds: [] },
  limits: { maxNodes: 100, maxRunSeconds: 3600, maxOutputBytes: 1048576 },
};

const TERMINAL_RUNS = new Set(["cancelled", "completed", "failed", "interrupted"]);

function statusVariant(status: string): "secondary" | "outline" | "destructive" {
  if (["failed", "interrupted", "rejected"].includes(status)) return "destructive";
  if (["completed", "approved"].includes(status)) return "secondary";
  return "outline";
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function documentFromRevision(revision: AgentGraphRevision): AgentGraphDocument {
  return {
    name: revision.name,
    description: revision.description,
    metadata: revision.metadata,
    inputSchema: revision.inputSchema,
    outputSchema: revision.outputSchema,
    nodes: revision.nodes,
    edges: revision.edges,
    permissions: revision.permissions,
    limits: revision.limits,
  };
}

export function AgentGraphsPanel({ projectId }: { projectId: string }) {
  const [graphs, setGraphs] = useState<AgentGraphSummary[]>([]);
  const [selectedGraph, setSelectedGraph] = useState<AgentGraphSummary | null>(null);
  const [documentText, setDocumentText] = useState(() => pretty(SAMPLE_GRAPH));
  const [inputText, setInputText] = useState('{\n  "task": "inspect this project"\n}');
  const [runs, setRuns] = useState<AgentGraphRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<AgentGraphRun | null>(null);
  const [runNodes, setRunNodes] = useState<AgentGraphNodeExecution[]>([]);
  const [runEvents, setRunEvents] = useState<AgentGraphEvent[]>([]);
  const [approvals, setApprovals] = useState<AgentGraphApproval[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadGraphs = useCallback(async () => {
    try {
      const next = await listAgentGraphs(projectId);
      setGraphs(next);
      setSelectedGraph((current) => {
        if (!current) return current;
        const replacement = next.find((graph) => graph.id === current.id) ?? null;
        if (
          replacement &&
          replacement.currentRevision === current.currentRevision &&
          replacement.updatedAt === current.updatedAt
        ) {
          return current;
        }
        return replacement;
      });
    } catch (reason) {
      setError(safeAgentWorkspaceError(reason));
    }
  }, [projectId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadGraphs(), 0);
    return () => window.clearTimeout(timer);
  }, [loadGraphs]);

  useEffect(() => {
    if (!selectedGraph) return;
    void listAgentGraphRuns(projectId, selectedGraph.id, 50)
      .then(setRuns)
      .catch((reason) => setError(safeAgentWorkspaceError(reason)));
    void getAgentGraph(projectId, selectedGraph.id)
      .then((result) => setDocumentText(pretty(documentFromRevision(result.revision))))
      .catch((reason) => setError(safeAgentWorkspaceError(reason)));
  }, [projectId, selectedGraph]);

  const refreshRun = useCallback(
    async (runId: string) => {
      const detail = await getAgentGraphRun(projectId, runId);
      const events = await listAgentGraphEvents(projectId, runId);
      setSelectedRun(detail.run);
      setRunNodes(detail.nodes);
      setApprovals(detail.approvals);
      setRunEvents(events);
      if (selectedGraph) setRuns(await listAgentGraphRuns(projectId, selectedGraph.id, 50));
    },
    [projectId, selectedGraph],
  );

  useEffect(() => {
    if (!selectedRun || TERMINAL_RUNS.has(selectedRun.status)) return;
    const timer = window.setInterval(() => {
      void refreshRun(selectedRun.id).catch((reason) => setError(safeAgentWorkspaceError(reason)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [refreshRun, selectedRun]);

  const activeApproval = useMemo(
    () => approvals.find((approval) => approval.status === "pending"),
    [approvals],
  );

  async function action<T>(key: string, work: () => Promise<T>, complete?: (value: T) => void) {
    if (busy) return;
    setBusy(key);
    setError(null);
    try {
      const result = await work();
      complete?.(result);
      toast.success("Graph updated");
    } catch (reason) {
      const message = safeAgentWorkspaceError(reason);
      setError(message);
      toast.error("Graph action failed", { description: message });
    } finally {
      setBusy(null);
    }
  }

  async function save() {
    let document: AgentGraphDocument;
    try {
      document = JSON.parse(documentText) as AgentGraphDocument;
    } catch {
      setError("Graph document is not valid JSON.");
      return;
    }
    await action("save", async () => {
      if (selectedGraph) {
        return updateAgentGraph(projectId, selectedGraph, document);
      }
      return createAgentGraph(projectId, document);
    }, async (graph) => {
      setSelectedGraph(graph);
      await loadGraphs();
    });
  }

  async function startRun() {
    if (!selectedGraph) {
      setError("Save a graph before starting a run.");
      return;
    }
    let input: Record<string, unknown>;
    try {
      input = JSON.parse(inputText) as Record<string, unknown>;
    } catch {
      setError("Run input is not valid JSON.");
      return;
    }
    await action("run", () => startAgentGraphRun(projectId, selectedGraph.id, { input }), (run) => {
      setSelectedRun(run);
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      void refreshRun(run.id);
    });
  }

  async function removeGraph() {
    if (!selectedGraph || !window.confirm(`Delete graph "${selectedGraph.name}" and its stopped run history?`)) return;
    const graphId = selectedGraph.id;
    await action("delete", async () => {
      await deleteAgentGraph(projectId, graphId);
    }, async () => {
      setSelectedGraph(null);
      setSelectedRun(null);
      setRuns([]);
      setRunNodes([]);
      setRunEvents([]);
      setApprovals([]);
      setDocumentText(pretty(SAMPLE_GRAPH));
      await loadGraphs();
    });
  }

  async function runMutation(
    operation: (projectId: string, runId: string) => Promise<AgentGraphRun>,
  ) {
    if (!selectedRun) return;
    await action("run-mutation", () => operation(projectId, selectedRun.id), (run) => {
      setSelectedRun(run);
      void refreshRun(run.id);
    });
  }

  async function decide(approval: AgentGraphApproval, decision: "approved" | "rejected") {
    if (!selectedRun) return;
    await action("approval", () => decideAgentGraphApproval(projectId, selectedRun.id, approval.id, decision), () => {
      void refreshRun(selectedRun.id);
    });
  }

  return (
    <section className="rounded-[22px] border border-border/60 bg-card/35 px-4 py-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <Play className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-ui-14 font-semibold text-foreground">Sloth graphs</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Versioned project workflows built from the existing Loop runtime.
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Loop and model nodes need a durable runtime selection with a model before they can run.
          </p>
        </div>
        <Button type="button" size="xs" variant="ghost" onClick={() => void loadGraphs()} disabled={Boolean(busy)}>
          <RefreshCw className={busy === "refresh" ? "animate-spin" : ""} /> Refresh
        </Button>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium">Project graphs</p>
            <Button type="button" size="xs" variant="outline" onClick={() => { setSelectedGraph(null); setDocumentText(pretty(SAMPLE_GRAPH)); }} disabled={Boolean(busy)}>
              <Plus /> New
            </Button>
          </div>
          {graphs.map((graph) => (
            <button
              type="button"
              key={graph.id}
              onClick={() => setSelectedGraph(graph)}
              className={`w-full rounded-xl px-3 py-2 text-left text-xs ${selectedGraph?.id === graph.id ? "bg-muted" : "bg-muted/35"}`}
            >
              <span className="block truncate font-medium">{graph.name}</span>
              <span className="text-[11px] text-muted-foreground">Revision {graph.currentRevision}</span>
            </button>
          ))}
          {graphs.length === 0 ? <p className="rounded-xl bg-muted/35 px-3 py-4 text-center text-xs text-muted-foreground">No graphs yet.</p> : null}
          {selectedGraph ? (
            <div className="rounded-xl border border-border/60 bg-background/45 p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-medium">Runs</p>
                <Button type="button" size="icon-xs" variant="ghost" onClick={() => void listAgentGraphRuns(projectId, selectedGraph.id, 50).then(setRuns)} aria-label="Refresh graph runs">
                  <RefreshCw />
                </Button>
              </div>
              <div className="mt-2 space-y-1.5">
                {runs.slice(0, 12).map((run) => (
                  <button type="button" key={run.id} onClick={() => void refreshRun(run.id)} className="flex w-full items-center gap-2 rounded-lg bg-muted/35 px-2.5 py-2 text-left text-[11px]">
                    <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
                    <span className="min-w-0 flex-1 truncate">{run.currentNodeId || run.id.slice(0, 8)}</span>
                    <span className="text-muted-foreground">r{run.revision}</span>
                  </button>
                ))}
                {runs.length === 0 ? <p className="text-[11px] text-muted-foreground">No runs yet.</p> : null}
              </div>
            </div>
          ) : null}
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-medium">Revision editor</p>
            <div className="flex items-center gap-1.5">
              {selectedGraph ? <Button type="button" size="xs" variant="ghost" onClick={() => void removeGraph()} disabled={Boolean(busy)}><Trash2 /> Delete</Button> : null}
              <Button type="button" size="xs" onClick={() => void save()} disabled={Boolean(busy)}>
                {busy === "save" ? <Loader2 className="animate-spin" /> : <CheckCircle2 />} Save revision
              </Button>
            </div>
          </div>
          <Textarea value={documentText} onChange={(event) => setDocumentText(event.target.value)} aria-label="Graph revision JSON" className="min-h-[300px] font-mono text-[11px]" spellCheck={false} />
          {selectedGraph ? <p className="text-[11px] text-muted-foreground">Saving creates revision {selectedGraph.currentRevision + 1}. Existing runs keep their pinned revision.</p> : <p className="text-[11px] text-muted-foreground">The first save creates revision 1. The backend validates IDs, edges, cycles, reachability, node configs, permissions, and budgets.</p>}
          <div className="rounded-xl border border-border/60 bg-background/45 p-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium">Run input</p>
              <Button type="button" size="xs" onClick={() => void startRun()} disabled={Boolean(busy) || !selectedGraph}>
                {busy === "run" ? <Loader2 className="animate-spin" /> : <Play />} Start run
              </Button>
            </div>
            <Textarea value={inputText} onChange={(event) => setInputText(event.target.value)} aria-label="Graph run input JSON" className="mt-2 min-h-20 font-mono text-[11px]" spellCheck={false} />
          </div>
          {selectedRun ? (
            <div className="rounded-xl border border-border/60 bg-background/45 p-3">
              <div className="flex items-center gap-2">
                <Badge variant={statusVariant(selectedRun.status)}>{selectedRun.status}</Badge>
                <span className="min-w-0 flex-1 truncate text-xs">Run {selectedRun.id}</span>
                {selectedRun.status === "paused" ? <Button type="button" size="icon-xs" variant="ghost" onClick={() => void runMutation(resumeAgentGraphRun)} aria-label="Resume graph run"><Play /></Button> : null}
                {["running", "pausing"].includes(selectedRun.status) ? <Button type="button" size="icon-xs" variant="ghost" onClick={() => void runMutation(pauseAgentGraphRun)} aria-label="Pause graph run"><Pause /></Button> : null}
                {["queued", "running", "pausing", "paused"].includes(selectedRun.status) ? <Button type="button" size="icon-xs" variant="ghost" onClick={() => void runMutation(cancelAgentGraphRun)} aria-label="Cancel graph run"><Square /></Button> : null}
                {["failed", "cancelled", "interrupted"].includes(selectedRun.status) ? <Button type="button" size="icon-xs" variant="ghost" onClick={() => void runMutation(retryAgentGraphRun)} aria-label="Retry graph run"><RotateCcw /></Button> : null}
              </div>
              {selectedRun.error ? <p className="mt-2 text-[11px] text-destructive">{selectedRun.error}</p> : null}
              {activeApproval ? (
                <div className="mt-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-[11px]">
                  <p className="font-medium">{activeApproval.title}</p>
                  {activeApproval.description ? <p className="mt-1 text-muted-foreground">{activeApproval.description}</p> : null}
                  <div className="mt-2 flex gap-1.5">
                    <Button type="button" size="xs" onClick={() => void decide(activeApproval, "approved")} disabled={Boolean(busy)}>Approve</Button>
                    <Button type="button" size="xs" variant="outline" onClick={() => void decide(activeApproval, "rejected")} disabled={Boolean(busy)}>Reject</Button>
                  </div>
                </div>
              ) : null}
              <details className="mt-2" open>
                <summary className="cursor-pointer text-[11px] font-medium">Node executions ({runNodes.length})</summary>
                <div className="mt-1 space-y-1">
                  {runNodes.map((node) => <div key={String(node.id)} className="flex items-center gap-2 rounded bg-muted/35 px-2 py-1.5 text-[11px]"><Badge variant={statusVariant(String(node.status))}>{String(node.status)}</Badge><span className="font-mono">{String(node.nodeId)}</span></div>)}
                </div>
              </details>
              <details className="mt-2">
                <summary className="cursor-pointer text-[11px] font-medium">Event log ({runEvents.length})</summary>
                <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/35 p-2 font-mono text-[10px] text-muted-foreground">{pretty(runEvents)}</pre>
              </details>
              {selectedRun.output !== null ? <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/35 p-2 font-mono text-[10px]">{pretty(selectedRun.output)}</pre> : null}
            </div>
          ) : null}
        </div>
      </div>
      {error ? <p className="mt-2 text-xs text-destructive">{error}</p> : null}
    </section>
  );
}
