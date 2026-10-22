import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { PolicyModal } from "../components/PolicyModal";
import { ReplayFeed, phaseNowLabel, renderedEvents } from "../components/ReplayFeed";

// Base step at 1× — one revealed event per interval; the speed multiplier divides it.
const PLAY_STEP_MS = 450;
const SPEEDS = [1, 2, 4, 8] as const;
type Speed = (typeof SPEEDS)[number];

// Playback uses the loaded reconstruction; these queries do not poll periodically.
export function ReplayReconstruct() {
  const { reviewId } = useParams<{ reviewId: string }>();
  const enabled = typeof reviewId === "string" && reviewId.length > 0;
  const pathParams = { params: { path: { review_id: reviewId ?? "" } } };

  const detailQuery = $api.useQuery("get", "/api/reviews/{review_id}", pathParams, { enabled });
  const timelineQuery = $api.useQuery("get", "/api/reviews/{review_id}/replay-timeline", pathParams, {
    enabled,
  });

  const reducedMotion =
    typeof window !== "undefined" &&
    Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

  const timelineData = timelineQuery.data;
  const review = detailQuery.data;
  const eventCount = useMemo(() => (timelineData ? renderedEvents(timelineData).length : 0), [timelineData]);
  // Finding events remain available when a divergent replay suppresses finding content.
  const findingCount = useMemo(
    () => (timelineData ? timelineData.events.filter((event) => event.event_type === "finding").length : 0),
    [timelineData],
  );
  const equivalent = Boolean(timelineData?.replay_equivalent && timelineData.phases !== null);

  const [revealedCount, setRevealedCount] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
  const [showPolicy, setShowPolicy] = useState(false);
  const [follow, setFollow] = useState(true);
  const feedRef = useRef<HTMLDivElement>(null);

  // Reduced motion and divergent reconstructions display the whole stream immediately.
  useEffect(() => {
    if (!timelineData) return;
    if (reducedMotion || !equivalent || eventCount === 0) {
      setRevealedCount(eventCount);
      setPlaying(false);
    } else {
      setRevealedCount(0);
      setPlaying(true);
    }
  }, [timelineData?.review_id, eventCount, equivalent, reducedMotion]);

  // Re-arm the one-event timer when playback state or speed changes.
  useEffect(() => {
    if (!playing) return;
    if (revealedCount >= eventCount) {
      setPlaying(false);
      return;
    }
    const playbackTimer = window.setTimeout(
      () => setRevealedCount((currentCount) => Math.min(currentCount + 1, eventCount)),
      PLAY_STEP_MS / speed,
    );
    return () => window.clearTimeout(playbackTimer);
  }, [playing, revealedCount, eventCount, speed]);

  // The final tick drops .current, so follow the last row only after reaching the end.
  useEffect(() => {
    if (!follow || !playing) return;
    const feedElement = feedRef.current;
    if (!feedElement) return;
    let playheadRow: HTMLElement | null = feedElement.querySelector(".tl-evrow.current");
    if (!playheadRow && revealedCount >= eventCount && eventCount > 0) {
      const eventRows = feedElement.querySelectorAll<HTMLElement>(".tl-evrow");
      playheadRow = eventRows[eventRows.length - 1] ?? null;
    }
    playheadRow?.scrollIntoView({ block: "nearest" });
  }, [revealedCount, follow, playing, eventCount]);

  // Manual scrolling stops following without interrupting playback.
  useEffect(() => {
    const stopFollowing = () => setFollow(false);
    window.addEventListener("wheel", stopFollowing, { passive: true });
    window.addEventListener("touchmove", stopFollowing, { passive: true });
    return () => {
      window.removeEventListener("wheel", stopFollowing);
      window.removeEventListener("touchmove", stopFollowing);
    };
  }, []);

  const play = () => {
    if (reducedMotion) {
      setRevealedCount(eventCount);
      return;
    }
    if (revealedCount >= eventCount) setRevealedCount(0); // restart from the top if already at the end
    setFollow(true);
    setPlaying(true);
  };
  const pause = () => setPlaying(false);
  const restart = () => {
    if (reducedMotion) {
      setRevealedCount(eventCount);
      return;
    }
    setFollow(true);
    setRevealedCount(0);
    setPlaying(true);
  };

  const playbackComplete = revealedCount >= eventCount && eventCount > 0;
  const progressPercent = eventCount > 0 ? Math.round((Math.min(revealedCount, eventCount) / eventCount) * 100) : 0;

  return (
    <section>
      <Link to={`/reviews/${reviewId}`} className="backlink">
        ← back to review
      </Link>

      <div className="panel">
        <div className="panel-b">
          <div className="rd-head">
            <div className="rd-title-block">
              <h1 className="rd-title">
                Replay · reconstruct{" "}
                {review ? (
                  <>
                    repo {review.repo_id} <span className="prnum">#{review.pr_number}</span>
                  </>
                ) : (
                  reviewId
                )}
              </h1>
              <div className="rd-meta">
                {review?.policy_version ? (
                  <button
                    type="button"
                    className="chip policy mono"
                    aria-haspopup="dialog"
                    aria-expanded={showPolicy}
                    onClick={() => setShowPolicy(true)}
                  >
                    policy {review.policy_version}
                  </button>
                ) : null}
                <span className="pill">🔒 append-only · read-only reconstruction</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-h">
          <h2>Real-time reconstruction</h2>
          <div className="sub">
            events replayed in execution order from the append-only log · nothing is edited
          </div>
          <div className="right rp-transport">
            <span className="rp-phase-now" aria-live="polite">
              {timelineData ? `phase ${phaseNowLabel(timelineData, revealedCount)}` : "—"}
            </span>
            <span className="rp-counter" aria-live="polite">
              event <b>{Math.min(revealedCount, eventCount)}</b> / <b>{eventCount}</b>
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
            <button
              type="button"
              className={`btn${follow ? " active" : ""}`}
              aria-pressed={follow}
              onClick={() => setFollow((following) => !following)}
              aria-label={follow ? "Stop following the playhead" : "Follow the playhead"}
            >
              {follow ? "⤓ Following" : "⤓ Follow"}
            </button>
            <span className="rp-speed" role="group" aria-label="Playback speed">
              {SPEEDS.map((speedOption) => (
                <button
                  key={speedOption}
                  type="button"
                  className={`rp-speed-btn ${speed === speedOption ? "active" : ""}`}
                  aria-pressed={speed === speedOption}
                  onClick={() => setSpeed(speedOption)}
                >
                  {speedOption}×
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
            aria-valuenow={progressPercent}
          >
            <div className="rp-progress-fill" style={{ width: `${progressPercent}%` }} />
          </div>
          {playbackComplete && timelineData ? (
            <>
              <div className="rp-verdict show" aria-live="polite">
                <span className="vmark" aria-hidden="true">
                  {equivalent ? "✓" : "✗"}
                </span>
                <span>
                  <b>{equivalent ? "replay-equivalent" : "not replay-equivalent"}</b> ·{" "}
                  <span className="vmeta">
                    {eventCount} events · {findingCount} findings
                    {timelineData.mode ? ` · ${timelineData.mode}` : ""}
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

      <div className="panel" ref={feedRef}>
        <div className="panel-h">
          <h2>Reconstructed audit feed</h2>
          <div className="sub">phase-grouped · {eventCount} events</div>
        </div>
        {timelineQuery.isLoading ? (
          <div className="panel-b">
            <p>Loading reconstruction…</p>
          </div>
        ) : timelineQuery.error ? (
          <div className="panel-b">
            <p className="error">Failed to load the reconstruction.</p>
          </div>
        ) : timelineData ? (
          <ReplayFeed data={timelineData} shown={revealedCount} />
        ) : null}
      </div>

      {showPolicy && review?.policy_version ? (
        <PolicyModal version={review.policy_version} onClose={() => setShowPolicy(false)} />
      ) : null}
    </section>
  );
}
