import { type FormEvent, type ReactNode, useState } from "react";

import { useTokenStore } from "./token";

/** Show the key form until a token is present; clearing it reopens the gate. */
export function TokenGate({ children }: { children: ReactNode }) {
  const token = useTokenStore((state) => state.token);
  const setToken = useTokenStore((state) => state.setToken);
  const [tokenInput, setTokenInput] = useState("");

  if (token) {
    return <>{children}</>;
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    const enteredToken = tokenInput.trim();
    if (enteredToken) {
      setToken(enteredToken);
    }
  };

  return (
    <div className="token-gate">
      <form className="token-gate__form" onSubmit={onSubmit}>
        <h1 className="token-gate__brand">Outrider</h1>
        <p className="token-gate__hint">Enter the operator admin API key.</p>
        <input
          type="password"
          value={tokenInput}
          onChange={(event) => setTokenInput(event.target.value)}
          placeholder="admin API key"
          aria-label="admin API key"
          autoFocus
        />
        <button type="submit" disabled={!tokenInput.trim()}>
          Enter
        </button>
      </form>
    </div>
  );
}
