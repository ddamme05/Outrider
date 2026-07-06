import {
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";

import type { components } from "../api/schema";
import {
  type AuditEvent,
  eventFamily,
  eventNode,
  isDiagnosticEvent,
  summarizeEvent,
} from "../lib/auditEvent";
import { formatDurationMs, spanMs } from "../lib/format";
import { hitlOutcomeLabel, severityLabel, typeLabel } from "../lib/findingSections";
import { AuditFeed } from "./AuditFeed";
import { CodeBlock } from "./CodeBlock";

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

// Rich per-event body for the flat `.ae` feed (mockup #screen-detail): the key fields bolded /
// mono'd per event type. Mirrors summarizeEvent's narrowing, reading ONLY fields the AuditEvent
// union actually carries — no fabricated latency / scope counts (display-only, DECISIONS#014/#016).
function flatBody(e: AuditEvent): ReactNode {
  switch (e.event_type) {
    case "llm_call":
      return (
        <>
          <b>{e.model}</b> · ${e.cost_usd.toFixed(2)} ·{" "}
          <span className="mono">{e.input_tokens}</span> in /{" "}
          <span className="mono">{e.output_tokens}</span> out
        </>
      );
    case "finding":
      return (
        <>
          <b>{severityLabel(e.severity)}</b> {typeLabel(e.finding_type)} ·{" "}
          <span className="mono">{(e.evidence_tier ?? "").toUpperCase()}</span> ·{" "}
          <span className="mono">
            {e.file_path}:{e.line_start}
          </span>
        </>
      );
    case "review_phase":
      return (
        <>
          <b>{e.node_id}</b> · marker <b>{e.marker}</b>
        </>
      );
    case "file_examination":
      return (
        <>
          examined <span className="mono">{e.file_path}</span>
        </>
      );
    case "trace_decision":
      return (
        <>
          <span className="mono">{e.resolution_status}</span>
        </>
      );
    case "agent_transition":
      return (
        <>
          node <b>{e.from_node}</b> → <b>{e.to_node}</b>
        </>
      );
    default:
      return summarizeEvent(e);
  }
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
        {(proof.tier ?? "").toUpperCase()}
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
          {content.evidence ? <CodeBlock code={content.evidence} /> : null}
          {content.suggested_fix ? (
            <div className="tl-c-fix">Fix: {content.suggested_fix}</div>
          ) : null}
          {h ? (
            // Override provenance from the HITLDecisionEvent stream (DECISIONS#034), not the table.
            <div className="f-prov">
              {hitlOutcomeLabel(h.outcome)}
              {h.original_severity && h.override_severity ? (
                <span className="prov-sev">
                  {" "}
                  · {severityLabel(h.original_severity)} → {severityLabel(h.override_severity)}
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
          {/* render a label+block only when the field is present — an empty <pre> under a
              "prompt"/"completion" label would imply absence-by-design rather than a null DB row
              (matches the truthy guard FindingContentPanel uses for evidence). */}
          {content.prompt ? (
            <>
              <div className="tl-c-label">prompt</div>
              <pre className="tl-c-pre">{content.prompt}</pre>
            </>
          ) : null}
          {content.completion ? (
            <>
              <div className="tl-c-label">completion</div>
              <pre className="tl-c-pre">{content.completion}</pre>
            </>
          ) : null}
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
  flat = false,
}: {
  data: TimelineData;
  shown?: number;
  // false (default) → collapsible phase CARDS (the Replay page, with progressive reveal).
  // true → the mockup's rich FLAT `.ae` feed (the review-detail static audit-of-record).
  flat?: boolean;
}) {
  const rendered = useMemo(() => renderedEvents(data), [data]);
  const total = rendered.length;
  // Shadow/telemetry events (cache_lookup, scope_exclusion, observed_skip_shadow) dominate the
  // analyze fan-out — one set per file. Hidden by default so the feed reads as review signal; a
  // toggle with the explicit count reveals them (nothing silently dropped — audit-completeness).
  const diagCount = useMemo(
    () => rendered.filter((e) => isDiagnosticEvent(e)).length,
    [rendered],
  );
  // Relative-timestamp baseline for the flat feed (client-computed — no server field): the
  // EARLIEST timestamp across the first phase-start marker AND every rendered event. Taking the
  // min (not phases[0].start alone) keeps relTime ≥ 0 for a leading inter-phase transition that
  // predates the first phase start — otherwise spanMs goes negative → null → a blank time column.
  const base = useMemo(() => {
    let min: string | null = data.phases?.[0]?.start?.timestamp ?? null;
    for (const e of rendered) {
      if (e.timestamp && (min === null || e.timestamp < min)) min = e.timestamp;
    }
    return min;
  }, [data.phases, rendered]);
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
  const [showDiag, setShowDiag] = useState(false);
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

  // Per-event content lookup + expand state, shared by the card row and the flat `.ae` row.
  const rowData = (e: AuditEvent) => {
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
    return { fc, lc, proof, id, expandable, open: expandable && expanded.has(id) };
  };

  // Relative timestamp for a flat-feed row, e.g. "+19s" / "+088ms" (mockup .ae-time).
  const relTime = (e: AuditEvent): string => {
    if (!base || !e.timestamp) return "";
    const ms = spanMs(base, e.timestamp);
    return ms === null ? "" : `+${formatDurationMs(Math.max(0, ms))}`;
  };

  const onRowKey = (id: string) => (ev: ReactKeyboardEvent) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      toggle(id);
    }
  };

  const row = (e: AuditEvent, withNode: boolean) => {
    const { fc, lc, proof, id, expandable, open } = rowData(e);
    return (
      <div key={e.event_id} className="tl-evgroup">
        <div
          className={`tl-evrow ev-c-${eventFamily(e.event_type)} ${cls(e)}${expandable ? " tl-expandable" : ""}`}
          role={expandable ? "button" : undefined}
          tabIndex={expandable ? 0 : undefined}
          aria-expanded={expandable ? open : undefined}
          onClick={expandable ? () => toggle(id) : undefined}
          onKeyDown={expandable ? onRowKey(id) : undefined}
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

  // The flat `.ae` row (mockup #screen-detail audit feed): a colored event-type chip, a rich body,
  // and a relative timestamp. Still expand-on-click for finding/llm rows (the PR-2 content panels) —
  // an enhancement over the static mockup, not a regression.
  const aeRow = (e: AuditEvent) => {
    const { fc, lc, proof, id, expandable, open } = rowData(e);
    return (
      <div key={e.event_id} className="tl-evgroup">
        <div
          className={`ae ev-c-${eventFamily(e.event_type)}${expandable ? " tl-expandable" : ""}`}
          role={expandable ? "button" : undefined}
          tabIndex={expandable ? 0 : undefined}
          aria-expanded={expandable ? open : undefined}
          onClick={expandable ? () => toggle(id) : undefined}
          onKeyDown={expandable ? onRowKey(id) : undefined}
        >
          <span className="ae-type mono">{e.event_type}</span>
          <span className="ae-body">
            {expandable ? <span aria-hidden="true">{open ? "▾ " : "▸ "}</span> : null}
            {flatBody(e)}
          </span>
          <span className="ae-time mono">{relTime(e)}</span>
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

  // Reveal control for the shadow/telemetry rows — shown only when there are any. Explicit
  // count keeps the audit complete: hidden, not dropped, and one click away.
  const diagToggle =
    diagCount > 0 ? (
      <button type="button" className="tl-diagtoggle" onClick={() => setShowDiag((v) => !v)}>
        {showDiag
          ? `Hide ${diagCount} diagnostic events`
          : `Show ${diagCount} hidden diagnostic events (cache · scope · OBSERVED shadow)`}
      </button>
    ) : null;

  // Flat feed (review-detail audit-of-record, mockup #screen-detail): append-only banner, then a
  // single CHRONOLOGICAL walk of the sequence-ordered stream. A `.ae-phase` divider is emitted when
  // a new phase's events begin; inter-phase transitions (e.g. analyze→trace — phase-unbounded, so
  // they fall outside every phase) render INLINE at their real position rather than dumped in one
  // block at the top. This preserves true append-only order (the banner's own claim) and matches
  // the mockup, which shows each transition at the tail of the phase it follows.
  if (flat) {
    const phaseByEventId = new Map<string, Phase>();
    for (const p of phases) {
      for (const e of p.events) {
        if (e.event_id) phaseByEventId.set(e.event_id, p);
      }
    }
    const feed: ReactNode[] = [];
    let currentPhaseId: string | null = null;
    const feedEvents = showDiag
      ? rendered
      : rendered.filter((e) => !isDiagnosticEvent(e));
    for (const e of feedEvents) {
      const ph = phaseByEventId.get(e.event_id ?? "") ?? null;
      if (ph && ph.phase_id !== currentPhaseId) {
        currentPhaseId = ph.phase_id;
        feed.push(
          <div className="ae-phase" key={`phase-${ph.phase_id}`}>
            <span className="pname">{ph.node_id}</span>
            <span className="pdur">{phaseDuration(ph)}</span>
            {ph.end === null ? <span className="pill">in-flight</span> : null}
          </div>,
        );
      }
      feed.push(aeRow(e));
    }
    return (
      <div className="panel-b">
        <div className="audit-note">
          <span className="lock" aria-hidden="true">
            🔒
          </span>
          Append-only by database policy — events cannot be edited or deleted; corrections append
          new events. This is what makes replay-equivalence verifiable.
        </div>
        {diagToggle}
        {feed}
      </div>
    );
  }

  return (
    <div className="panel-b">
      {diagToggle}
      {data.inter_phase_events.length > 0 ? (
        <div className="tl-inter">
          <div className="dist-sub-h">between phases</div>
          {data.inter_phase_events.map((e) => row(e, false))}
        </div>
      ) : null}

      {phases.map((phase) => {
        const evs = showDiag
          ? phase.events
          : phase.events.filter((e) => !isDiagnosticEvent(e));
        return (
          <div key={phase.phase_id} className="tl-phase">
            <div className="tl-phase-head">
              <span className="tl-node mono">{phase.node_id}</span>
              <span className="tl-dur mono">{phaseDuration(phase)}</span>
              {phase.end === null ? <span className="pill">in-flight</span> : null}
            </div>
            {evs.length === 0 ? (
              <div className="tl-empty">no operations</div>
            ) : (
              evs.map((e) => row(e, true))
            )}
          </div>
        );
      })}
    </div>
  );
}
