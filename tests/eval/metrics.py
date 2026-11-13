"""Define saved evaluation metrics; scoring lives in grading.py and scorecard.py.

Ratio metrics retain their counts as well as the ratio. Cost and latency
fields name their units. Invalid ranges and unexpected fields are rejected."""

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
