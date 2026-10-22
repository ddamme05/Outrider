import { useEffect, type ReactNode } from "react";

// Dialog callers supply content; this shell owns backdrop/Escape closing and labels.
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
    const handleKeyDown = (keyEvent: KeyboardEvent) => {
      if (keyEvent.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      onClick={onClose}
    >
      <div className="modal" onClick={(clickEvent) => clickEvent.stopPropagation()}>
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
