import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { $api } from "../api/client";
import type { components } from "../api/schema";
import {
  type DecisionDraft,
  EMPTY_DRAFT,
  RESUME_WINDOW_MS,
  isDraftValid,
  toPayload,
} from "../lib/hitl";

type FindingView = components["schemas"]["FindingView"];

export function useReviewApproval({
  actionable,
  gatedFindings,
  productionConfirmed,
}: {
  actionable: boolean;
  gatedFindings: readonly FindingView[];
  productionConfirmed: boolean;
}) {
  const queryClient = useQueryClient();
  const [drafts, setDrafts] = useState<Record<string, DecisionDraft>>({});
  const [submitted, setSubmitted] = useState(false);
  const decisionMutation = $api.useMutation("post", "/reviews/{review_id}/decide");
  const [submitSeq, setSubmitSeq] = useState(0);
  const [stuck, setStuck] = useState(false);
  // Retries must resend the accepted payload. Changed decisions conflict with its
  // audit record, so the route keeps editing locked until a reload.
  const [submittedPayload, setSubmittedPayload] = useState<ReturnType<typeof toPayload>[] | null>(
    null,
  );

  // Each accepted submit starts a new window for detecting a stalled resume.
  useEffect(() => {
    if (submitSeq === 0) return;
    setStuck(false);
    const resumeTimer = setTimeout(() => setStuck(true), RESUME_WINDOW_MS);
    return () => clearTimeout(resumeTimer);
  }, [submitSeq]);

  const getDraft = (findingId: string): DecisionDraft => drafts[findingId] ?? EMPTY_DRAFT;
  const setDraft = (findingId: string, nextDraft: DecisionDraft) =>
    setDrafts((currentDrafts) => ({ ...currentDrafts, [findingId]: nextDraft }));
  const decidedCount = gatedFindings.filter(
    (finding) => isDraftValid(getDraft(finding.finding_id), finding),
  ).length;
  const allGatedValid = gatedFindings.length > 0 && decidedCount === gatedFindings.length;
  const resumed = submitted && !actionable;
  const canSubmit =
    productionConfirmed &&
    actionable &&
    allGatedValid &&
    !decisionMutation.isPending &&
    (!submitted || stuck);

  const submit = (reviewId: string) => {
    const decisions =
      submittedPayload ??
      gatedFindings.map((finding) => toPayload(getDraft(finding.finding_id), finding));
    decisionMutation.mutate(
      { params: { path: { review_id: reviewId } }, body: { decisions } },
      {
        onSuccess: () => {
          setSubmitted(true);
          setSubmittedPayload(decisions);
          setSubmitSeq((sequence) => sequence + 1);
          // The 202 resumes work in the background. Polling tracks its progress.
          void queryClient.invalidateQueries();
        },
      },
    );
  };

  return {
    getDraft,
    setDraft,
    decidedCount,
    submitted,
    stuck,
    resumed,
    canSubmit,
    isPending: decisionMutation.isPending,
    error: decisionMutation.error,
    submit,
  };
}
