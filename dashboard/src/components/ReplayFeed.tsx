import { useEffect, useMemo, useState } from "react";

import type { components } from "../api/schema";
import { type AuditEvent, eventFamily, eventNode, summarizeEvent } from "../lib/auditEvent";
import { formatDurationMs, spanMs } from "../lib/format";
import { AuditFeed } from "./AuditFeed";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
type Phase = NonNullable<TimelineData["phases"]>[number];
type FindingContent = TimelineData["findings"][number];
type LLMContent = TimelineData["llm_exchanges"][number];

// The finding's proof artifacts — evidence tier + query_match_id (OBSERVED) / trace_path (INFERRED).
// These ride the permanent FindingEvent (not the purgeable content row), so they render even on a
// redacted stub — the "metadata is permanent" half of DECISIONS#014/#016 / the proof boundary.
type ProofMeta = { tier: string; queryMatchId: string | null; tracePath: string[] | null };

// The rows the GROUPED view actually renders, in chronological order: the flat stream MINUS the
// phase start/end markers (the phase CARDS represent those — they are never event rows). Shared by
// ReplayFeed (the body) and the replay page (its progress/counter/phase-now), so the playback
// denominator always matches what's on screen.
export function renderedEvents(data: TimelineData): AuditEvent[] {
  if (data.phases === null) return [...data.events];
  const markerIds = new Set<string>();
  for (const p of data.phases) {
    if (p.start?.event_id) markerIds.add(p.start.event_id);
    if (p.end?.event_id) markerIds.add(p.end.event_id);
  }
  return data.events.filter((e) => !markerIds.has(e.event_id ?? ""));
}

// The node_id of the phase containing the most-recently-revealed event (`shown`-th rendered row),
// for the replay page's "phase <node>" indicator. "—" before playback, "between phases" for an
// inter-phase row.
export function phaseNowLabel(data: TimelineData, shown: number): string {
  const rendered = renderedEvents(data);
  if (shown <= 0 || rendered.length === 0) return "—";
  const cur = rendered[Math.min(shown, rendered.length) - 1];
  for (const p of data.phases ?? []) {
    if (p.events.some((e) => e.event_id === cur?.event_id)) return p.node_id;
  }
  return "between phases";
}

// The duration slot stays empty for an open phase — the in-flight pill conveys that state.
function phaseDuration(phase: Phase): string {
  const ms = spanMs(phase.start?.timestamp, phase.end?.timestamp);
  return ms === null ? "—" : formatDurationMs(ms);
}

// Retention-redacted stub — content purged, metadata permanent (DECISIONS#014/#016, the same
// shape FindingCard renders). The redaction date is the per-table-per-sweep `purge_audit` time.
function RedactedNote({ kind, sweepAt }: { kind: "finding" | "llm"; sweepAt: string | null }) {
  const permanent = kind === "finding" ? "type, severity, location, proof" : "model, tokens, cost";
  const purged = kind === "finding" ? "title/description/evidence/fix" : "prompt/response";
  return (
    <div className="f-desc redacted">
      Content redacted{sweepAt ? ` in the retention sweep on ${sweepAt.slice(0, 10)}` : ""}. The
      {kind === "finding" ? " finding's" : " call's"} metadata ({permanent}) is permanent; its{" "}
      {purged} were purged per the retention policy.
    </div>
  );
}

function FindingContentPanel({ content, proof }: { content: FindingContent; proof: ProofMeta }) {
  const h = content.hitl_decision;
  return (
    <div className="tl-content">
      {/* Proof artifacts render unconditionally — they survive content redaction. */}
      <div className="tl-c-proof mono">
        {proof.tier}
        {proof.queryMatchId ? <span> · {proof.queryMatchId}</span> : null}
        {proof.tracePath && proof.tracePath.length > 0 ? (
          <span> · trace {proof.tracePath.join(" → ")}</span>
        ) : null}
      </div>
      {content.content_redacted ? (
        <RedactedNote kind="finding" sweepAt={content.redaction_sweep_at} />
      ) : (
        <>
          {content.title ? <div className="tl-c-title">{content.title}</div> : null}
          {content.description ? <div className="tl-c-body">{content.description}</div> : null}
          {content.evidence ? <pre className="tl-c-pre">{content.evidence}</pre> : null}
          {content.suggested_fix ? (
            <div className="tl-c-fix">Fix: {content.suggested_fix}</div>
          ) : null}
          {h ? (
            // Override provenance from the HITLDecisionEvent stream (DECISIONS#034), not the table.
            <div className="f-prov">
              {h.outcome}
              {h.original_severity && h.override_severity ? (
                <span className="prov-sev">
                  {" "}
                  · {h.original_severity} → {h.override_severity}
                </span>
              ) : null}
              <span className="prov-by"> · by {h.reviewer_id}</span>
              {h.reason ? <span> · {h.reason}</span> : null}
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}

function LLMContentPanel({ content }: { content: LLMContent }) {
  return (
    <div className="tl-content">
      {content.content_redacted ? (
        <RedactedNote kind="llm" sweepAt={content.redaction_sweep_at} />
      ) : (
        <>
          <div className="tl-c-label">prompt</div>
          <pre className="tl-c-pre">{content.prompt}</pre>
          <div className="tl-c-label">completion</div>
          <pre className="tl-c-pre">{content.completion}</pre>
        </>
      )}
    </div>
  );
}

// The phase-grouped reconstruction body — phase cards with per-operation event rows that expand on
// click to the content `findings`/`llm_exchanges` carry (redacted-stub when purged). `shown` drives
// progressive reveal: rows with rendered-index >= shown are hidden (`future`), the row at shown-1
// fades in (`current`); the default (Infinity) reveals everything (the review-detail static feed).
// A non-equivalent verdict (phases===null) degrades to the flat ordered feed + a banner (FUP-125).
export function ReplayFeed({
  data,
  shown = Number.POSITIVE_INFINITY,
}: {
  data: TimelineData;
  shown?: number;
}) {
  const rendered = useMemo(() => renderedEvents(data), [data]);
  const total = rendered.length;
  const orderIndex = useMemo(
    () => new Map(rendered.map((e, i) => [e.event_id, i])),
    [rendered],
  );
  const findingsByFid = useMemo(
    () => new Map(data.findings.map((f) => [f.finding_id, f])),
    [data.findings],
  );
  const llmByEventId = useMemo(
    () => new Map(data.llm_exchanges.map((x) => [x.event_id, x])),
    [data.llm_exchanges],
  );

  // Per-row expand state, keyed by event_id — independent of playback, cleared only on a review
  // change (so an open panel survives the 2s poll / a play step).
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  useEffect(() => setExpanded(new Set()), [data.review_id]);
  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // "" (revealed / resting), "current" (the just-revealed row, fades in), "future" (not yet
  // reached — hidden during progressive playback).
  const cls = (e: AuditEvent): string => {
    if (shown >= total) return "";
    const i = orderIndex.get(e.event_id) ?? 0;
    if (i === shown - 1) return "current";
    return i >= shown ? "future" : "";
  };

  const row = (e: AuditEvent, withNode: boolean) => {
    const fc = e.event_type === "finding" ? findingsByFid.get(e.finding_id) : undefined;
    const lc = e.event_type === "llm_call" ? llmByEventId.get(e.event_id ?? "") : undefined;
    // Proof artifacts ride the permanent FindingEvent (in `events`/`phases`), not the content view.
    const proof: ProofMeta | null =
      e.event_type === "finding"
        ? {
            tier: e.evidence_tier,
            queryMatchId: e.query_match_id ?? null,
            tracePath: e.trace_path ?? null,
          }
        : null;
    const id = e.event_id ?? "";
    const expandable = fc !== undefined || lc !== undefined;
    const open = expandable && expanded.has(id);
    return (
      <div key={e.event_id} className="tl-evgroup">
        <div
          className={`tl-evrow ev-c-${eventFamily(e.event_type)} ${cls(e)}${expandable ? " tl-expandable" : ""}`}
          role={expandable ? "button" : undefined}
          tabIndex={expandable ? 0 : undefined}
          aria-expanded={expandable ? open : undefined}
          onClick={expandable ? () => toggle(id) : undefined}
          onKeyDown={
            expandable
              ? (ev) => {
                  if (ev.key === "Enter" || ev.key === " ") {
                    ev.preventDefault();
                    toggle(id);
                  }
                }
              : undefined
          }
        >
          <span className="af-type mono">
            {expandable ? (open ? "▾ " : "▸ ") : ""}
            {e.event_type}
          </span>
          {withNode ? <span className="af-node mono">{eventNode(e) ?? ""}</span> : null}
          <span className="af-summary">{summarizeEvent(e)}</span>
        </div>
        {open && fc && proof ? <FindingContentPanel content={fc} proof={proof} /> : null}
        {open && lc ? <LLMContentPanel content={lc} /> : null}
      </div>
    );
  };

  if (!data.replay_equivalent || data.phases === null) {
    return (
      <div className="panel-b">
        <p className="queue-notice" role="alert">
          Not replay-equivalent — the phase grouping is unavailable.
          {data.reason ? <span className="mono"> {data.reason}</span> : null}
        </p>
        <AuditFeed events={data.events} />
      </div>
    );
  }

  const phases = data.phases;
  return (
    <div className="panel-b">
      {data.inter_phase_events.length > 0 ? (
        <div className="tl-inter">
          <div className="dist-sub-h">between phases</div>
          {data.inter_phase_events.map((e) => row(e, false))}
        </div>
      ) : null}

      {phases.map((phase) => (
        <div key={phase.phase_id} className="tl-phase">
          <div className="tl-phase-head">
            <span className="tl-node mono">{phase.node_id}</span>
            <span className="tl-dur mono">{phaseDuration(phase)}</span>
            {phase.end === null ? <span className="pill">in-flight</span> : null}
          </div>
          {phase.events.length === 0 ? (
            <div className="tl-empty">no operations</div>
          ) : (
            phase.events.map((e) => row(e, true))
          )}
        </div>
      ))}
    </div>
  );
}
