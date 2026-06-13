import { useEffect } from "react";

// Static explainer for the replay-equivalent verdict, opened by clicking the
// verdict pill in the review header (mirrors the PolicyModal pattern). No
// fetch — the concept is fixed; the per-review ✓/✗ verdict lives on the pill.
// Closes on backdrop click or Escape.
export function ReplayInfoModal({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="What replay-equivalent means"
      onClick={onClose}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-h">
          <div>
            <h2>Replay equivalence</h2>
            <div className="sub">Every review is reconstructable from its append-only audit log.</div>
          </div>
          <button type="button" className="modal-close" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-b">
          <p className="modal-note">
            Outrider records every step of a review — each LLM call, file examined, finding, and
            human decision — as an append-only event stream. <strong>Replay</strong> reconstructs the
            review from that stream alone and compares the result against what was originally
            produced.
          </p>
          <p className="modal-note">
            <strong style={{ color: "var(--pos)" }}>replay-equivalent ✓</strong> — the reconstruction
            matched the original review exactly. The audit log is a faithful, independently
            verifiable record of what the agent did.
          </p>
          <p className="modal-note">
            <strong style={{ color: "var(--neg)" }}>not replay-equivalent ✗</strong> — the
            reconstruction diverged from the original. A signal to investigate, since the stored
            record and the replay disagree.
          </p>
          <p className="modal-note">
            After the retention window the content (prompts, finding text) is redacted, but the event
            sequence is permanent — so the order of what happened stays provable even once the
            details can no longer be shown.
          </p>
        </div>
      </div>
    </div>
  );
}
