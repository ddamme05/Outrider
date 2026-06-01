import { useParams } from "react-router";

export function ReviewDetail() {
  const { reviewId } = useParams();
  return (
    <section>
      <h1 className="mono">Review {reviewId}</h1>
      <p style={{ color: "var(--text-2)" }}>
        Pipeline strip, metrics, the Findings tab, and the Replay verdict land here next.
      </p>
    </section>
  );
}
