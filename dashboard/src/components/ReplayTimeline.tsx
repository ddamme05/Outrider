import { useEffect, useMemo, useRef, useState } from "react";

import type { components } from "../api/schema";
import { type AuditEvent, eventFamily, eventNode, summarizeEvent } from "../lib/auditEvent";
import { AuditFeed } from "./AuditFeed";

type TimelineData = components["schemas"]["ReplayTimelineResponse"];
type Phase = NonNullable<TimelineData["phases"]>[number];

function phaseDuration(phase: Phase): string {
  if (phase.start?.timestamp && phase.end?.timestamp) {
    const ms = new Date(phase.end.timestamp).getTime() - new Date(phase.start.timestamp).getTime();
    if (Number.isFinite(ms) && ms >= 0) {
      return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
    }
  }
  return phase.end === null ? "in-flight" : "—";
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

// A read-only playable view of a review's reconstructed audit stream (ROADMAP feature 6,
// PR 1). The grouped `phases` come from the server's replay-VERIFIED reconstruction and are
// present only on an equivalent verdict (FUP-125); a non-equivalent review degrades to the
// flat ordered feed + a banner. Metadata-only — no content expansion (PR 2). Playback is
// pure client-side stepping over the static ordered DTO; nothing is fabricated.
export function ReplayTimeline({ data }: { data: TimelineData }) {
  const events = data.events;
  const total = events.length;
  // `shown` = how many ordered events have "played" (0..total); total = the resting full view.
  const [shown, setShown] = useState(total);
  const [playing, setPlaying] = useState(false);

  // event_id → flat-order index, so per-event playback state works across the grouped view.
  const orderIndex = useMemo(
    () => new Map(events.map((e, i) => [e.event_id, i])),
    [events],
  );

  // Reset playback when the review (its event set) changes.
  useEffect(() => {
    setShown(total);
    setPlaying(false);
  }, [total, data.review_id]);

  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    if (shown >= total) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => setShown((s) => Math.min(s + 1, total)), 450);
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, [playing, shown, total]);

  // "" (resting full view), else past / current / future relative to the play cursor.
  const cls = (e: AuditEvent): string => {
    if (shown >= total) return "";
    const i = orderIndex.get(e.event_id) ?? 0;
    if (i < shown - 1) return "past";
    if (i === shown - 1) return "current";
    return "future";
  };

  const row = (e: AuditEvent, withNode: boolean) => (
    <div key={e.event_id} className={`tl-evrow ev-c-${eventFamily(e.event_type)} ${cls(e)}`}>
      <span className="af-type mono">{e.event_type}</span>
      {withNode ? <span className="af-node mono">{eventNode(e) ?? ""}</span> : null}
      <span className="af-summary">{summarizeEvent(e)}</span>
    </div>
  );

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
