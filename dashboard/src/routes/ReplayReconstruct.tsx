import { useMemo, useState } from "react";
import { Link, useParams } from "react-router";

import { $api } from "../api/client";
import { PolicyModal } from "../components/PolicyModal";
import { ReplayFeed, phaseNowLabel, renderedEvents } from "../components/ReplayFeed";
import { PLAYBACK_SPEEDS, useReplayPlayback } from "./useReplayPlayback";

// Playback uses the loaded reconstruction; these queries do not poll periodically.
export function ReplayReconstruct() {
  const { reviewId } = useParams<{ reviewId: string }>();
  const enabled = typeof reviewId === "string" && reviewId.length > 0;
  const pathParams = { params: { path: { review_id: reviewId ?? "" } } };

  const detailQuery = $api.useQuery("get", "/api/reviews/{review_id}", pathParams, { enabled });
  const timelineQuery = $api.useQuery("get", "/api/reviews/{review_id}/replay-timeline", pathParams, {
    enabled,
  });

  const timelineData = timelineQuery.data;
  const review = detailQuery.data;
  const eventCount = useMemo(() => (timelineData ? renderedEvents(timelineData).length : 0), [timelineData]);
  // Finding events remain available when a divergent replay suppresses finding content.
  const findingCount = useMemo(
    () => (timelineData ? timelineData.events.filter((event) => event.event_type === "finding").length : 0),
    [timelineData],
  );
  const equivalent = Boolean(timelineData?.replay_equivalent && timelineData.phases !== null);

  const [showPolicy, setShowPolicy] = useState(false);
  const playback = useReplayPlayback({ timelineData, eventCount, equivalent });

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
              {timelineData ? `phase ${phaseNowLabel(timelineData, playback.revealedCount)}` : "—"}
            </span>
            <span className="rp-counter" aria-live="polite">
              event <b>{Math.min(playback.revealedCount, eventCount)}</b> / <b>{eventCount}</b>
            </span>
            <button
              type="button"
              className="btn"
              onClick={playback.play}
              disabled={playback.playing}
              aria-label="Play reconstruction"
            >
              ▶ Play
            </button>
            <button
              type="button"
              className="btn"
              onClick={playback.pause}
              disabled={!playback.playing}
              aria-label="Pause reconstruction"
            >
              ⏸ Pause
            </button>
            <button type="button" className="btn" onClick={playback.restart} aria-label="Restart reconstruction">
              ↻ Restart
            </button>
            <button
              type="button"
              className={`btn${playback.follow ? " active" : ""}`}
              aria-pressed={playback.follow}
              onClick={playback.toggleFollow}
              aria-label={playback.follow ? "Stop following the playhead" : "Follow the playhead"}
            >
              {playback.follow ? "⤓ Following" : "⤓ Follow"}
            </button>
            <span className="rp-speed" role="group" aria-label="Playback speed">
              {PLAYBACK_SPEEDS.map((speedOption) => (
                <button
                  key={speedOption}
                  type="button"
                  className={`rp-speed-btn ${playback.speed === speedOption ? "active" : ""}`}
                  aria-pressed={playback.speed === speedOption}
                  onClick={() => playback.setSpeed(speedOption)}
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
            aria-valuenow={playback.progressPercent}
          >
            <div className="rp-progress-fill" style={{ width: `${playback.progressPercent}%` }} />
          </div>
          {playback.complete && timelineData ? (
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

      <div className="panel" ref={playback.feedRef}>
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
          <ReplayFeed data={timelineData} shown={playback.revealedCount} />
        ) : null}
      </div>

      {showPolicy && review?.policy_version ? (
        <PolicyModal version={review.policy_version} onClose={() => setShowPolicy(false)} />
      ) : null}
    </section>
  );
}
