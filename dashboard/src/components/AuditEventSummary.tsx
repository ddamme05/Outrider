import { Link } from "react-router";

import { type AuditEvent, summarizeEvent } from "../lib/auditEvent";

export type CacheServeEvent = Extract<AuditEvent, { event_type: "cache_serve" }>;

export function OriginalReviewLink({ event }: { event: CacheServeEvent }) {
  const search = new URLSearchParams({ cache: event.cache_key });
  return <Link to={`/reviews/${event.source_review_id}?${search}`}>Original review for {event.file_path}</Link>;
}

export function AuditEventSummary({ event }: { event: AuditEvent }) {
  if (event.event_type === "cache_serve") {
    const count = event.served_finding_count;
    return (
      <>
        <OriginalReviewLink event={event} /> · {count} {count === 1 ? "finding reused" : "findings reused"}
      </>
    );
  }
  return summarizeEvent(event);
}
