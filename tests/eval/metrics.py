"""V1 eval-metric Pydantic shapes per spec §11.3.

Six metrics ship as Pydantic models in V1:

  - `FindingPrecision` — % of agent findings matching ground truth
  - `FindingRecall` — % of ground truth findings the agent identifies
  - `SeverityAccuracy` — % of findings where severity matches ground truth
  - `FalsePositiveRate` — % of findings clearly wrong (noise)
  - `CostPerReview` — summed from `LLMCallEvent.cost_usd`
  - `LatencyPerReview` — wall-clock from webhook receipt to review posted

The shapes only — the SCORING functions (LLM-as-judge for precision /
recall; deterministic comparison for severity accuracy and false-positive
rate) are non-goals of the eval-harness spec; they land with the
analyze-node spec when the LLM Protocol + provider wrappers exist.

Each metric carries:
  - `value: float` in `[0.0, 1.0]` for ratio metrics, or non-negative for
    cost/latency.
  - `numerator: int | None` and `denominator: int | None` for ratios so
    the dashboard can render "12 of 15" alongside the percentage.
  - Unit-bearing absolute metrics (`CostPerReview.usd`,
    `LatencyPerReview.seconds`) carry the unit in the field name to make
    the dashboard rendering unambiguous.

All metrics use `ConfigDict(extra="forbid")` per docs/conventions.md.
"""

from pydantic import BaseModel, ConfigDict, Field


class FindingPrecision(BaseModel):
    """Ratio: of N agent findings, how many match ground truth?"""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)


class FindingRecall(BaseModel):
    """Ratio: of N ground-truth findings, how many did the agent identify?"""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)


class SeverityAccuracy(BaseModel):
    """Ratio: of N matched findings, how many have severity matching ground truth?"""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)


class FalsePositiveRate(BaseModel):
    """Ratio: of N agent findings, how many are clearly wrong?"""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)


class CostPerReview(BaseModel):
    """Absolute USD cost per review, summed from `LLMCallEvent.cost_usd`."""

    model_config = ConfigDict(extra="forbid")

    usd: float = Field(ge=0.0)


class LatencyPerReview(BaseModel):
    """Absolute wall-clock seconds from webhook receipt to review posted."""

    model_config = ConfigDict(extra="forbid")

    seconds: float = Field(ge=0.0)


__all__ = [
    "CostPerReview",
    "FalsePositiveRate",
    "FindingPrecision",
    "FindingRecall",
    "LatencyPerReview",
    "SeverityAccuracy",
]
