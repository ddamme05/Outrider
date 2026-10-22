import { Modal } from "./Modal";

// Static explanation; the review header owns the per-review verdict.
export function ReplayInfoModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal
      ariaLabel="What replay-equivalent means"
      title="Replay equivalence"
      sub="Every review is reconstructable from its append-only audit log."
      onClose={onClose}
    >
      <p className="modal-note">
        Outrider records every step of a review — each LLM call, file examined, finding, and human
        decision — as an append-only event stream. <strong>Replay</strong> reconstructs the review
        from that stream alone and compares the result against what was originally produced.
      </p>
      <p className="modal-note">
        <strong style={{ color: "var(--pos)" }}>replay-equivalent ✓</strong> — the reconstruction
        matched the original review exactly. The audit log is a faithful, independently verifiable
        record of what the agent did.
      </p>
      <p className="modal-note">
        <strong style={{ color: "var(--neg)" }}>not replay-equivalent ✗</strong> — the
        reconstruction diverged from the original. A signal to investigate, since the stored record
        and the replay disagree.
      </p>
      <p className="modal-note">
        After the retention window the content (prompts, finding text) is redacted, but the event
        sequence is permanent — so the order of what happened stays provable even once the details
        can no longer be shown.
      </p>
    </Modal>
  );
}
