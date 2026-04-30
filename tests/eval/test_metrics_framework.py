"""Eval-harness metrics framework: 6 V1 metric Pydantic shapes per spec §11.3.

Pure schema test — constructs each metric with valid data and confirms the
shape rejects malformed inputs (negative values, ratios outside [0, 1],
extra fields).

The shapes only — scoring functions are non-goals here per the eval-harness
spec.
"""

import pytest
from pydantic import ValidationError

from .metrics import (
    CostPerReview,
    FalsePositiveRate,
    FindingPrecision,
    FindingRecall,
    LatencyPerReview,
    SeverityAccuracy,
)


def test_finding_precision_admits_valid_ratio() -> None:
    metric = FindingPrecision(value=0.8, numerator=12, denominator=15)
    assert metric.value == 0.8
    assert metric.numerator == 12
    assert metric.denominator == 15


def test_finding_precision_rejects_value_above_one() -> None:
    with pytest.raises(ValidationError):
        FindingPrecision(value=1.1, numerator=11, denominator=10)


def test_finding_precision_rejects_negative_numerator() -> None:
    with pytest.raises(ValidationError):
        FindingPrecision(value=0.5, numerator=-1, denominator=2)


def test_finding_precision_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FindingPrecision(  # type: ignore[call-arg]
            value=0.5,
            numerator=1,
            denominator=2,
            unknown_field="oops",
        )


def test_finding_recall_admits_zero_ratio() -> None:
    metric = FindingRecall(value=0.0, numerator=0, denominator=10)
    assert metric.value == 0.0


def test_severity_accuracy_admits_perfect_ratio() -> None:
    metric = SeverityAccuracy(value=1.0, numerator=10, denominator=10)
    assert metric.value == 1.0


def test_severity_accuracy_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        SeverityAccuracy(value=-0.1, numerator=0, denominator=10)


def test_false_positive_rate_admits_valid_ratio() -> None:
    metric = FalsePositiveRate(value=0.2, numerator=2, denominator=10)
    assert metric.value == 0.2


def test_cost_per_review_admits_zero_usd() -> None:
    metric = CostPerReview(usd=0.0)
    assert metric.usd == 0.0


def test_cost_per_review_rejects_negative_usd() -> None:
    with pytest.raises(ValidationError):
        CostPerReview(usd=-0.01)


def test_latency_per_review_admits_positive_seconds() -> None:
    metric = LatencyPerReview(seconds=42.5)
    assert metric.seconds == 42.5


def test_latency_per_review_rejects_negative_seconds() -> None:
    with pytest.raises(ValidationError):
        LatencyPerReview(seconds=-1.0)
