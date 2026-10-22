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
import { prettyModel } from "../lib/modelLabel";
import { AuditFeed } from "./AuditFeed";
import { CodeBlock } from "./CodeBlock";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
type Phase = NonNullable<TimelineData["phases"]>[number];
type FindingContent = TimelineData["findings"][number];
type LLMContent = TimelineData["llm_exchanges"][number];

// Proof comes from permanent finding events, independent of retained content.
type ProofMeta = { tier: string; queryMatchId: string | null; tracePath: string[] | null };

// Phase cards replace start/end markers. Playback counts all remaining events,
// including diagnostics that the visibility toggle may hide.
export function renderedEvents(data: TimelineData): AuditEvent[] {
  if (data.phases === null) return [...data.events];
  const markerIds = new Set<string>();
  for (const phase of data.phases) {
    if (phase.start?.event_id) markerIds.add(phase.start.event_id);
    if (phase.end?.event_id) markerIds.add(phase.end.event_id);
  }
  return data.events.filter((event) => !markerIds.has(event.event_id ?? ""));
}

// Name the phase of the latest revealed event, including inter-phase transitions.
export function phaseNowLabel(data: TimelineData, shown: number): string {
  const playbackEvents = renderedEvents(data);
  if (shown <= 0 || playbackEvents.length === 0) return "—";
  const currentEvent = playbackEvents[Math.min(shown, playbackEvents.length) - 1];
  for (const phase of data.phases ?? []) {
    if (phase.events.some((event) => event.event_id === currentEvent?.event_id)) return phase.node_id;
  }
  return "between phases";
}

// Open phases show an unknown duration alongside their in-flight badge.
function phaseDuration(phase: Phase): string {
  const durationMs = spanMs(phase.start?.timestamp, phase.end?.timestamp);
  return durationMs === null ? "—" : formatDurationMs(durationMs);
}

function flatEventBody(event: AuditEvent): ReactNode {
  switch (event.event_type) {
    case "llm_call":
      return (
        <>
          <b>{prettyModel(event.model)}</b> ·{" "}
          {event.cost_usd == null ? "unpriced" : `$${event.cost_usd.toFixed(2)}`} ·{" "}
          <span className="mono">{event.input_tokens}</span> in /{" "}
          <span className="mono">{event.output_tokens}</span> out
        </>
      );
    case "finding":
      return (
        <>
          <b>{severityLabel(event.severity)}</b> {typeLabel(event.finding_type)} ·{" "}
          <span className="mono">{(event.evidence_tier ?? "").toUpperCase()}</span> ·{" "}
          <span className="mono">
            {event.file_path}:{event.line_start}
          </span>
        </>
      );
    case "review_phase":
      return (
        <>
          <b>{event.node_id}</b> · marker <b>{event.marker}</b>
        </>
      );
    case "file_examination":
      return (
        <>
          examined <span className="mono">{event.file_path}</span>
        </>
      );
    case "trace_decision":
      return (
        <>
          <span className="mono">{event.resolution_status}</span>
        </>
      );
    case "agent_transition":
      return (
        <>
          node <b>{event.from_node}</b> → <b>{event.to_node}</b>
        </>
      );
    default:
      return summarizeEvent(event);
  }
}

// The redaction timestamp belongs to the table's retention sweep.
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
  const decision = content.hitl_decision;
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
          {decision ? (
            // Decisions are reconstructed from the permanent audit stream.
            <div className="f-prov">
              {hitlOutcomeLabel(decision.outcome)}
              {decision.original_severity && decision.override_severity ? (
                <span className="prov-sev">
                  {" "}
                  · {severityLabel(decision.original_severity)} → {severityLabel(decision.override_severity)}
                </span>
              ) : null}
              <span className="prov-by"> · by {decision.reviewer_id}</span>
              {decision.reason ? <span> · {decision.reason}</span> : null}
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
          {/* Missing content must not look like a recorded empty exchange. */}
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

// Expandable review events, rendered as grouped playback or a chronological flat feed.
export function ReplayFeed({
  data,
  shown = Number.POSITIVE_INFINITY,
  flat = false,
}: {
  data: TimelineData;
  shown?: number;
  // Flat is the static review-detail view; grouped cards support progressive playback.
  flat?: boolean;
}) {
  const playbackEvents = useMemo(() => renderedEvents(data), [data]);
  const eventCount = playbackEvents.length;
  // Fallback and grouped inter-phase rows retain their full event stream.
  const diagnosticCount = useMemo(
    () => playbackEvents.filter((event) => isDiagnosticEvent(event)).length,
    [playbackEvents],
  );
  // Include leading inter-phase events when finding the relative-time baseline.
  const startTimestamp = useMemo(() => {
    let earliestTimestamp: string | null = data.phases?.[0]?.start?.timestamp ?? null;
    for (const event of playbackEvents) {
      if (event.timestamp && (earliestTimestamp === null || event.timestamp < earliestTimestamp)) {
        earliestTimestamp = event.timestamp;
      }
    }
    return earliestTimestamp;
  }, [data.phases, playbackEvents]);
  const eventOrder = useMemo(
    () => new Map(playbackEvents.map((event, eventIndex) => [event.event_id, eventIndex])),
    [playbackEvents],
  );
  const findingsById = useMemo(
    () => new Map(data.findings.map((findingContent) => [findingContent.finding_id, findingContent])),
    [data.findings],
  );
  const llmContentByEventId = useMemo(
    () => new Map(data.llm_exchanges.map((exchange) => [exchange.event_id, exchange])),
    [data.llm_exchanges],
  );

  // Expanded panels survive polling and playback; only changing reviews resets them.
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  useEffect(() => setExpanded(new Set()), [data.review_id]);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const toggle = (id: string) =>
    setExpanded((previousExpanded) => {
      const nextExpanded = new Set(previousExpanded);
      if (nextExpanded.has(id)) nextExpanded.delete(id);
      else nextExpanded.add(id);
      return nextExpanded;
    });

  const playbackClass = (event: AuditEvent): string => {
    if (shown >= eventCount) return "";
    const eventIndex = eventOrder.get(event.event_id) ?? 0;
    if (eventIndex === shown - 1) return "current";
    return eventIndex >= shown ? "future" : "";
  };

  const eventContent = (event: AuditEvent) => {
    const findingContent = event.event_type === "finding" ? findingsById.get(event.finding_id) : undefined;
    const llmContent = event.event_type === "llm_call" ? llmContentByEventId.get(event.event_id ?? "") : undefined;
    const proof: ProofMeta | null =
      event.event_type === "finding"
        ? {
            tier: event.evidence_tier,
            queryMatchId: event.query_match_id ?? null,
            tracePath: event.trace_path ?? null,
          }
        : null;
    const id = event.event_id ?? "";
    const expandable = findingContent !== undefined || llmContent !== undefined;
    return { findingContent, llmContent, proof, id, expandable, open: expandable && expanded.has(id) };
  };

  const relativeTime = (event: AuditEvent): string => {
    if (!startTimestamp || !event.timestamp) return "";
    const durationMs = spanMs(startTimestamp, event.timestamp);
    return durationMs === null ? "" : `+${formatDurationMs(Math.max(0, durationMs))}`;
  };

  const onRowKeyDown = (id: string) => (keyEvent: ReactKeyboardEvent) => {
    if (keyEvent.key === "Enter" || keyEvent.key === " ") {
      keyEvent.preventDefault();
      toggle(id);
    }
  };

  const renderPhaseRow = (event: AuditEvent, withNode: boolean) => {
    const { findingContent, llmContent, proof, id, expandable, open } = eventContent(event);
    return (
      <div key={event.event_id} className="tl-evgroup">
        <div
          className={`tl-evrow ev-c-${eventFamily(event.event_type)} ${playbackClass(event)}${expandable ? " tl-expandable" : ""}`}
          role={expandable ? "button" : undefined}
          tabIndex={expandable ? 0 : undefined}
          aria-expanded={expandable ? open : undefined}
          onClick={expandable ? () => toggle(id) : undefined}
          onKeyDown={expandable ? onRowKeyDown(id) : undefined}
        >
          <span className="af-type mono">
            {expandable ? (open ? "▾ " : "▸ ") : ""}
            {event.event_type}
          </span>
          {withNode ? <span className="af-node mono">{eventNode(event) ?? ""}</span> : null}
          <span className="af-summary">{summarizeEvent(event)}</span>
        </div>
        {open && findingContent && proof ? <FindingContentPanel content={findingContent} proof={proof} /> : null}
        {open && llmContent ? <LLMContentPanel content={llmContent} /> : null}
      </div>
    );
  };

  const renderFlatRow = (event: AuditEvent) => {
    const { findingContent, llmContent, proof, id, expandable, open } = eventContent(event);
    return (
      <div key={event.event_id} className="tl-evgroup">
        <div
          className={`ae ev-c-${eventFamily(event.event_type)}${expandable ? " tl-expandable" : ""}`}
          role={expandable ? "button" : undefined}
          tabIndex={expandable ? 0 : undefined}
          aria-expanded={expandable ? open : undefined}
          onClick={expandable ? () => toggle(id) : undefined}
          onKeyDown={expandable ? onRowKeyDown(id) : undefined}
        >
          <span className="ae-type mono">{event.event_type}</span>
          <span className="ae-body">
            {expandable ? <span aria-hidden="true">{open ? "▾ " : "▸ "}</span> : null}
            {flatEventBody(event)}
          </span>
          <span className="ae-time mono">{relativeTime(event)}</span>
        </div>
        {open && findingContent && proof ? <FindingContentPanel content={findingContent} proof={proof} /> : null}
        {open && llmContent ? <LLMContentPanel content={llmContent} /> : null}
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

  const diagnosticToggle =
    diagnosticCount > 0 ? (
      <button type="button" className="tl-diagtoggle" onClick={() => setShowDiagnostics((v) => !v)}>
        {showDiagnostics
          ? `Hide ${diagnosticCount} diagnostic events`
          : `Show ${diagnosticCount} hidden diagnostic events (cache · scope · OBSERVED shadow)`}
      </button>
    ) : null;

  if (flat) return renderFlatFeed();

  return (
    <div className="panel-b">
      {diagnosticToggle}
      {data.inter_phase_events.length > 0 ? (
        <div className="tl-inter">
          <div className="dist-sub-h">between phases</div>
          {data.inter_phase_events.map((event) => renderPhaseRow(event, false))}
        </div>
      ) : null}

      {phases.map((phase) => {
        const visibleEvents = showDiagnostics
          ? phase.events
          : phase.events.filter((event) => !isDiagnosticEvent(event));
        return (
          <div key={phase.phase_id} className="tl-phase">
            <div className="tl-phase-head">
              <span className="tl-node mono">{phase.node_id}</span>
              <span className="tl-dur mono">{phaseDuration(phase)}</span>
              {phase.end === null ? <span className="pill">in-flight</span> : null}
            </div>
            {visibleEvents.length === 0 ? (
              <div className="tl-empty">no operations</div>
            ) : (
              visibleEvents.map((event) => renderPhaseRow(event, true))
            )}
          </div>
        );
      })}
    </div>
  );

  function renderFlatFeed() {
    // Walk the ordered stream once so transitions stay between the phases they connect.
    const phaseByEventId = new Map<string, Phase>();
    for (const phase of phases) {
      for (const event of phase.events) {
        if (event.event_id) phaseByEventId.set(event.event_id, phase);
      }
    }
    const feed: ReactNode[] = [];
    let currentPhaseId: string | null = null;
    const feedEvents = showDiagnostics
      ? playbackEvents
      : playbackEvents.filter((event) => !isDiagnosticEvent(event));
    for (const event of feedEvents) {
      const phase = phaseByEventId.get(event.event_id ?? "") ?? null;
      if (phase && phase.phase_id !== currentPhaseId) {
        currentPhaseId = phase.phase_id;
        feed.push(
          <div className="ae-phase" key={`phase-${phase.phase_id}`}>
            <span className="pname">{phase.node_id}</span>
            <span className="pdur">{phaseDuration(phase)}</span>
            {phase.end === null ? <span className="pill">in-flight</span> : null}
          </div>,
        );
      }
      feed.push(renderFlatRow(event));
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
        {diagnosticToggle}
        {feed}
      </div>
    );
  }
}
