import { type FormEvent, type ReactNode, useState } from "react";

import { useTokenStore } from "./token";

/** Renders `children` once an admin key is present; otherwise a paste-key gate.
 * Re-prompts automatically when the key is cleared (e.g. on a 401). */
export function TokenGate({ children }: { children: ReactNode }) {
  const token = useTokenStore((s) => s.token);
  const setToken = useTokenStore((s) => s.setToken);
  const [value, setValue] = useState("");

  if (token) {
    return <>{children}</>;
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = value.trim();
    if (trimmed) {
      setToken(trimmed);
    }
  };

  return (
    <div className="token-gate">
      <form className="token-gate__form" onSubmit={onSubmit}>
        <h1 className="token-gate__brand">Outrider</h1>
        <p className="token-gate__hint">Enter the operator admin API key.</p>
        <input
          type="password"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="admin API key"
          aria-label="admin API key"
          autoFocus
        />
        <button type="submit" disabled={!value.trim()}>
          Enter
        </button>
      </form>
    </div>
  );
}
