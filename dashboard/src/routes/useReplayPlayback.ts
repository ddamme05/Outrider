import { useEffect, useRef, useState } from "react";

import type { components } from "../api/schema";

const PLAY_STEP_MS = 450;
export const PLAYBACK_SPEEDS = [1, 2, 4, 8] as const;
type Speed = (typeof PLAYBACK_SPEEDS)[number];

export function useReplayPlayback({
  timelineData,
  eventCount,
  equivalent,
}: {
  timelineData: Pick<components["schemas"]["ReplayTimelineResponse"], "review_id"> | undefined;
  eventCount: number;
  equivalent: boolean;
}) {
  const reducedMotion =
    typeof window !== "undefined" &&
    Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

  const [revealedCount, setRevealedCount] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
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

  // The feed drops .current on the final tick. Follow the last row in that case.
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
    if (revealedCount >= eventCount) setRevealedCount(0);
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
  const toggleFollow = () => setFollow((following) => !following);

  const complete = revealedCount >= eventCount && eventCount > 0;
  const progressPercent =
    eventCount > 0 ? Math.round((Math.min(revealedCount, eventCount) / eventCount) * 100) : 0;

  return {
    revealedCount,
    playing,
    speed,
    setSpeed,
    follow,
    toggleFollow,
    feedRef,
    play,
    pause,
    restart,
    complete,
    progressPercent,
  };
}
