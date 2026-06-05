import { useEffect, useMemo, useState } from "react";

import type { components } from "../api/schema";
import { type AuditEvent, eventFamily, eventNode, summarizeEvent } from "../lib/auditEvent";
import { formatDurationMs, spanMs } from "../lib/format";
import { AuditFeed } from "./AuditFeed";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
type Phase = NonNullable<TimelineData["phases"]>[number];
type FindingContent = TimelineData["findings"][number];
type LLMContent = TimelineData["llm_exchanges"][number];

// The duration slot stays empty for an open phase — the in-flight pill conveys that state,
// so it isn't repeated here as text.
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
    <div className="tl-content f-desc redacted">
      Content redacted{sweepAt ? ` in the retention sweep on ${sweepAt.slice(0, 10)}` : ""}. The
      {kind === "finding" ? " finding's" : " call's"} metadata ({permanent}) is permanent; its{" "}
      {purged} were purged per the retention policy.
    </div>
  );
}

function FindingContentPanel({ content }: { content: FindingContent }) {
  if (content.content_redacted) {
    return <RedactedNote kind="finding" sweepAt={content.redaction_sweep_at} />;
  }
  const h = content.hitl_decision;
  return (
    <div className="tl-content">
      {content.title ? <div className="tl-c-title">{content.title}</div> : null}
      {content.description ? <div className="tl-c-body">{content.description}</div> : null}
      {content.evidence ? <pre className="tl-c-pre">{content.evidence}</pre> : null}
      {content.suggested_fix ? <div className="tl-c-fix">Fix: {content.suggested_fix}</div> : null}
      {h ? (
        // Override provenance from the HITLDecisionEvent stream (DECISIONS#034), never the table.
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
    </div>
  );
}

function LLMContentPanel({ content }: { content: LLMContent }) {
  if (content.content_redacted) {
    return <RedactedNote kind="llm" sweepAt={content.redaction_sweep_at} />;
  }
  return (
    <div className="tl-content">
      <div className="tl-c-label">prompt</div>
      <pre className="tl-c-pre">{content.prompt}</pre>
      <div className="tl-c-label">completion</div>
      <pre className="tl-c-pre">{content.completion}</pre>
    </div>
  );
}

function VerdictHeader({ data }: { data: TimelineData }) {
  const ok = data.replay_equivalent;
  return (
    <div className="panel-h">
      <h2>Replay timeline</h2>
      <div className="sub">
        <span className={`pill ${ok ? "" : "status-expired"}`} aria-label="replay verdict">
          {ok ? "✓ replay-equivalent" : "✗ not replay-equivalent"}
        </span>
        {data.mode ? <span className="mono"> · {data.mode}</span> : null}
        {data.status ? <span className="mono"> · {data.status}</span> : null}
      </div>
    </div>
  );
}

// A read-only playable view of a review's reconstructed audit stream (ROADMAP feature 6).
// The grouped `phases` come from the server's replay-VERIFIED reconstruction and are present
// only on an equivalent verdict (FUP-125); a non-equivalent review degrades to the flat ordered
// feed + a banner. `finding` / `llm_call` rows expand on click to show the content `findings` /
// `llm_exchanges` carry (PR 2) — redacted-stub when purged. Playback is pure client-side stepping
// over the static ordered DTO; nothing is fabricated.
export function ReplayTimeline({ data }: { data: TimelineData }) {
  const events = data.events;
  // The rows the GROUPED view actually renders, in chronological order: the flat stream MINUS
  // the phase start/end markers (the phase CARDS represent those — they are never event rows).
  // Playback counts + steps over THESE, so the scrubber denominator and the play cursor match
  // what's on screen; counting raw `events` would over-count by ~2 markers/phase and stall the
  // cursor on invisible marker steps. (Non-equivalent → no scrubber renders; `rendered` is moot.)
  const rendered = useMemo(() => {
    if (data.phases === null) return events;
    const markerIds = new Set<string>();
    for (const p of data.phases) {
      if (p.start?.event_id) markerIds.add(p.start.event_id);
      if (p.end?.event_id) markerIds.add(p.end.event_id);
    }
    return events.filter((e) => !markerIds.has(e.event_id ?? ""));
  }, [events, data.phases]);
  const total = rendered.length;
  // `shown` = how many ordered rows have "played" (0..total); total = the resting full view.
  const [shown, setShown] = useState(total);
  const [playing, setPlaying] = useState(false);

  // event_id → rendered-order index, so per-event playback state works across the grouped view.
  const orderIndex = useMemo(
    () => new Map(rendered.map((e, i) => [e.event_id, i])),
    [rendered],
  );

  // Content lookups for the expand panels: findings by finding_id, LLM exchanges by event_id.
  const findingsByFid = useMemo(
    () => new Map(data.findings.map((f) => [f.finding_id, f])),
    [data.findings],
  );
  const llmByEventId = useMemo(
    () => new Map(data.llm_exchanges.map((x) => [x.event_id, x])),
    [data.llm_exchanges],
  );

  // Per-row expand state, keyed by event_id — independent of playback (`shown`/`orderIndex`) and
  // NOT reset by the 2s poll (only a review change clears it), so an open panel survives refetches.
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  useEffect(() => setExpanded(new Set()), [data.review_id]);
  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  // Reset playback when the review (its rendered set) changes.
  useEffect(() => {
    setShown(total);
    setPlaying(false);
  }, [total, data.review_id]);

  useEffect(() => {
    if (!playing) return;
    if (shown >= total) {
      setPlaying(false);
      return;
    }
    // Capture the handle locally so the cleanup clears THIS effect's timer, not a later one.
    const handle = window.setTimeout(() => setShown((s) => Math.min(s + 1, total)), 450);
    return () => window.clearTimeout(handle);
  }, [playing, shown, total]);

  // "" (resting full view OR an already-played row — neither is styled), "current" at the
  // play cursor, "future" for not-yet-played rows (dimmed).
  const cls = (e: AuditEvent): string => {
    if (shown >= total) return "";
    const i = orderIndex.get(e.event_id) ?? 0;
    if (i === shown - 1) return "current";
    return i >= shown ? "future" : "";
  };

  const row = (e: AuditEvent, withNode: boolean) => {
    const fc = e.event_type === "finding" ? findingsByFid.get(e.finding_id) : undefined;
    const lc = e.event_type === "llm_call" ? llmByEventId.get(e.event_id ?? "") : undefined;
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
        {open && fc ? <FindingContentPanel content={fc} /> : null}
        {open && lc ? <LLMContentPanel content={lc} /> : null}
      </div>
    );
  };

  // FUP-125: phases are trustworthy only on an equivalent verdict. Otherwise the grouping is
  // suppressed server-side (`phases === null`); render the flat ordered stream + a banner.
  if (!data.replay_equivalent || data.phases === null) {
    return (
      <div className="panel">
        <VerdictHeader data={data} />
        <div className="panel-b">
          <p className="queue-notice" role="alert">
            Not replay-equivalent — the phase grouping is unavailable.
            {data.reason ? <span className="mono"> {data.reason}</span> : null}
          </p>
          <AuditFeed events={events} />
        </div>
      </div>
    );
  }

  const phases = data.phases;
  return (
    <div className="panel">
      <VerdictHeader data={data} />
      <div className="panel-b">
        <div className="tl-scrub">
          <button
            type="button"
            className="btn"
            onClick={() => {
              setShown(0);
              setPlaying(true);
            }}
          >
            ▶ Play
          </button>
          <button type="button" className="btn" onClick={() => setPlaying(false)}>
            ⏸ Pause
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setPlaying(false);
              setShown((s) => Math.max(0, s - 1));
            }}
          >
            ◀
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setPlaying(false);
              setShown((s) => Math.min(total, s + 1));
            }}
          >
            ▶
          </button>
          <input
            type="range"
            min={0}
            max={total}
            value={Math.min(shown, total)}
            aria-label="timeline position"
            onChange={(ev) => {
              setPlaying(false);
              setShown(Number(ev.target.value));
            }}
          />
          <span className="mono tl-pos">
            {Math.min(shown, total)}/{total}
          </span>
        </div>

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
    </div>
  );
}
