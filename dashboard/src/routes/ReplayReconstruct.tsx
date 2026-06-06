import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { ReplayFeed, phaseNowLabel, renderedEvents } from "../components/ReplayFeed";

// Base step at 1× — one revealed event per interval; the speed multiplier divides it.
const PLAY_STEP_MS = 450;
const SPEEDS = [1, 2, 4, 8] as const;
type Speed = (typeof SPEEDS)[number];

// The dedicated real-time reconstruction page (ROADMAP feature 6, "replay-equivalence is something a
// buyer can watch"). Reached from the review's "Replay · reconstruct" button. Events replay in
// execution order from the verified /replay-timeline snapshot — append-only, read-only; nothing is
// edited or fabricated. NOT polled: the operator replays a fixed reconstruction, not a moving target.
export function ReplayReconstruct() {
  const { reviewId } = useParams<{ reviewId: string }>();
  const enabled = typeof reviewId === "string" && reviewId.length > 0;
  const pathParams = { params: { path: { review_id: reviewId ?? "" } } };

  const detail = $api.useQuery("get", "/api/reviews/{review_id}", pathParams, { enabled });
  const timeline = $api.useQuery("get", "/api/reviews/{review_id}/replay-timeline", pathParams, {
    enabled,
  });

  const reducedMotion =
    typeof window !== "undefined" &&
    Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

  const data = timeline.data;
  const d = detail.data;
  const total = useMemo(() => (data ? renderedEvents(data).length : 0), [data]);
  // Count finding EVENTS off the permanent stream, not data.findings — the content array is
  // suppressed to empty on a non-equivalent verdict (gated in reviews.py), so reading its length
  // would fabricate "0 findings" for a divergent review that actually produced findings. The
  // FindingEvents ride data.events in both cases (reviews.py builds events regardless of equivalence).
  const findingCount = useMemo(
    () => (data ? data.events.filter((e) => e.event_type === "finding").length : 0),
    [data],
  );
  const equivalent = Boolean(data?.replay_equivalent && data.phases !== null);

  const [shown, setShown] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);

  // Auto-play once the snapshot loads — or render instantly under reduced motion / a non-equivalent
  // verdict (where the flat fallback shows everything and there's nothing to step through).
  useEffect(() => {
    if (!data) return;
    if (reducedMotion || !equivalent || total === 0) {
      setShown(total);
      setPlaying(false);
    } else {
      setShown(0);
      setPlaying(true);
    }
  }, [data?.review_id, total, equivalent, reducedMotion]);

  // The play timer: reveal one more row every PLAY_STEP_MS / speed. Re-armed when speed changes.
  useEffect(() => {
    if (!playing) return;
    if (shown >= total) {
      setPlaying(false);
      return;
    }
    const handle = window.setTimeout(
      () => setShown((s) => Math.min(s + 1, total)),
      PLAY_STEP_MS / speed,
    );
    return () => window.clearTimeout(handle);
  }, [playing, shown, total, speed]);

  const play = () => {
    if (reducedMotion) {
      setShown(total);
      return;
    }
    if (shown >= total) setShown(0); // restart from the top if already at the end
    setPlaying(true);
  };
  const pause = () => setPlaying(false);
  const restart = () => {
    if (reducedMotion) {
      setShown(total);
      return;
    }
    setShown(0);
    setPlaying(true);
  };

  const atRest = shown >= total && total > 0;
  const pct = total > 0 ? Math.round((Math.min(shown, total) / total) * 100) : 0;

  return (
    <section>
      <Link to={`/reviews/${reviewId}`} className="backlink">
        ← back to review
      </Link>

      {/* header */}
      <div className="panel">
        <div className="panel-b">
          <div className="rd-head">
            <div className="rd-title-block">
              <h1 className="rd-title">
                Replay · reconstruct{" "}
                {d ? (
                  <>
                    repo {d.repo_id} <span className="prnum">#{d.pr_number}</span>
                  </>
                ) : (
                  reviewId
                )}
              </h1>
              <div className="rd-meta">
                {d?.policy_version ? <span className="pill mono">policy {d.policy_version}</span> : null}
                <span className="pill">🔒 append-only · read-only reconstruction</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* transport + progress */}
      <div className="panel">
        <div className="panel-h">
          <h2>Real-time reconstruction</h2>
          <div className="sub">
            events replayed in execution order from the append-only log · nothing is edited
          </div>
          <div className="right rp-transport">
            <span className="rp-phase-now" aria-live="polite">
              {data ? `phase ${phaseNowLabel(data, shown)}` : "—"}
            </span>
            <span className="rp-counter" aria-live="polite">
              event <b>{Math.min(shown, total)}</b> / <b>{total}</b>
            </span>
            <button type="button" className="btn" onClick={play} disabled={playing} aria-label="Play reconstruction">
              ▶ Play
            </button>
            <button type="button" className="btn" onClick={pause} disabled={!playing} aria-label="Pause reconstruction">
              ⏸ Pause
            </button>
            <button type="button" className="btn" onClick={restart} aria-label="Restart reconstruction">
              ↻ Restart
            </button>
            <span className="rp-speed" role="group" aria-label="Playback speed">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className={`rp-speed-btn ${speed === s ? "active" : ""}`}
                  aria-pressed={speed === s}
                  onClick={() => setSpeed(s)}
                >
                  {s}×
                </button>
              ))}
            </span>
          </div>
        </div>
        <div className="panel-b">
          <div
            className="rp-progress"
            role="progressbar"
            aria-label="Reconstruction progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
          >
            <div className="rp-progress-fill" style={{ width: `${pct}%` }} />
          </div>
          {atRest && data ? (
            <>
              <div className="rp-verdict show" aria-live="polite">
                <span className="vmark" aria-hidden="true">
                  {equivalent ? "✓" : "✗"}
                </span>
                <span>
                  <b>{equivalent ? "replay-equivalent" : "not replay-equivalent"}</b> ·{" "}
                  <span className="vmeta">
                    {total} events · {findingCount} findings
                    {data.mode ? ` · ${data.mode}` : ""}
                  </span>
                </span>
              </div>
              <div className="rp-retain">
                <span className="lock">🔒</span>
                Within the retention window, full content (LLM exchanges + finding text) reconstructs
                from <span className="mono">audit_events</span> + content tables. After retention, the
                event stream alone supports <b>metadata-only</b> replay — the sequence is permanent;
                content fields are redacted per policy.
              </div>
            </>
          ) : null}
        </div>
      </div>

      {/* reconstructed feed */}
      <div className="panel">
        <div className="panel-h">
          <h2>Reconstructed audit feed</h2>
          <div className="sub">phase-grouped · {total} events</div>
        </div>
        {timeline.isLoading ? (
          <div className="panel-b">
            <p>Loading reconstruction…</p>
          </div>
        ) : timeline.error ? (
          <div className="panel-b">
            <p className="error">Failed to load the reconstruction.</p>
          </div>
        ) : data ? (
          <ReplayFeed data={data} shown={shown} />
        ) : null}
      </div>
    </section>
  );
}
