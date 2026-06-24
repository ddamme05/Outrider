"""V1 eval-metric Pydantic shapes per spec §11.3.

Six metrics ship as Pydantic models in V1:

  - `FindingPrecision` — % of agent findings matching ground truth
  - `FindingRecall` — % of ground truth findings the agent identifies
  - `SeverityAccuracy` — % of findings where severity matches ground truth
  - `FalsePositiveRate` — % of findings clearly wrong (noise)
  - `CostPerReview` — summed from `LLMCallEvent.cost_usd`
  - `LatencyPerReview` — wall-clock from webhook receipt to review posted

The shapes only — this module defines no scoring. Deterministic scoring
SHIPPED in `grading.py` (recall / precision / severity-accuracy via a
structural match contract — deliberately NOT LLM-as-judge): `grading.py`
consumes `FindingPrecision`/`FindingRecall`/`SeverityAccuracy`, and the eval
scorecard (`scorecard.py` / `runner.py`) consumes `FalsePositiveRate` /
`CostPerReview` / `LatencyPerReview`. The earlier "LLM-as-judge scorer lands
with the analyze-node spec" plan was rejected, not built.

Each metric carries:
  - `value: float` in `[0.0, 1.0]` for ratio metrics, or non-negative for
    cost/latency.
  - `numerator: int` and `denominator: int` (both `ge=0`) for ratios so
    the dashboard can render "12 of 15" alongside the percentage. Both
    required, not optional — for "of N findings, how many ..." shapes,
    both N and the count are well-defined integers by definition.
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
