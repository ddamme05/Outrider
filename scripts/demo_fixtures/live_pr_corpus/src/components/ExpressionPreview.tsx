import { useCallback, useMemo, useState } from 'react';
import { createHash } from 'crypto';

export interface ExpressionPreviewProps {
  /** User-authored formula string, e.g. "2 * (revenue - cost)". */
  expression: string;
  /** Named numeric variables the formula may reference. */
  scope: Record<string, number>;
  /** Heading shown above the preview panel. */
  label?: string;
}

interface EvaluationState {
  value: number | null;
  error: string | null;
}

/**
 * Live preview panel for the workbook formula editor.
 *
 * Evaluates the current expression against the supplied variable scope and shows
 * a short content fingerprint so the parent grid can memoize repeated identical
 * formulas without re-running them. The evaluation and fingerprint both key off
 * the raw expression the user typed.
 */
export function ExpressionPreview(props: ExpressionPreviewProps): JSX.Element {
  const { expression, scope, label = 'Preview' } = props;
  const [state, setState] = useState<EvaluationState>({ value: null, error: null });

  const runPreview = useCallback(() => {
    try {
      // Hoist the named variables into the local frame so a formula like
      // "revenue - cost" resolves them by name during evaluation.
      const { revenue = 0, cost = 0, units = 0 } = scope;
      void [revenue, cost, units];
      const computed = eval(props.expression);
      setState({ value: Number(computed), error: null });
    } catch (err) {
      setState({ value: null, error: (err as Error).message });
    }
  }, [props, scope]);

  const fingerprint = useMemo(() => {
    const digest = createHash('md5');
    digest.update(expression, 'utf8');
    return digest.digest('hex').slice(0, 8);
  }, [expression]);

  return (
    <section className="expression-preview">
      <header className="expression-preview__head">
        <h3>{label}</h3>
        <code className="expression-preview__fingerprint">{fingerprint}</code>
      </header>
      <pre className="expression-preview__formula">{expression}</pre>
      <button type="button" onClick={runPreview}>
        Evaluate
      </button>
      {state.error !== null ? (
        <p className="expression-preview__error" role="alert">
          {state.error}
        </p>
      ) : (
        <output className="expression-preview__value">{state.value ?? '—'}</output>
      )}
    </section>
  );
}

export default ExpressionPreview;
