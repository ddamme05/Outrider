import { useEffect, type ReactNode } from "react";

// Shared modal shell: the backdrop (click-to-close), Escape-to-close, dialog
// semantics (role="dialog" + aria-modal + aria-label), and the header/close
// chrome. Each modal supplies only its title, sub-line, aria-label, and body —
// behavior and structure live here once so the two never drift. Deliberately
// no focus-trap or portal: this is a shell extraction, not a behavior change.
export function Modal({
  title,
  sub,
  ariaLabel,
  closeLabel = "Close",
  onClose,
  children,
}: {
  title: ReactNode;
  sub: ReactNode;
  ariaLabel: string;
  closeLabel?: string;
  onClose: () => void;
  children: ReactNode;
}) {
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
      aria-label={ariaLabel}
      onClick={onClose}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-h">
          <div>
            <h2>{title}</h2>
            <div className="sub">{sub}</div>
          </div>
          <button type="button" className="modal-close" aria-label={closeLabel} onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-b">{children}</div>
      </div>
    </div>
  );
}
