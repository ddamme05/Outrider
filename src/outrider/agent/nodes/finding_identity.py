"""Distinguish co-reported judged claims before recording their candidate receipts."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from outrider.audit.events import compute_finding_content_hash
from outrider.policy.findings import EvidenceTier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.review_finding import ReviewFinding


def _claim_groups(findings: list[ReviewFinding]) -> dict[tuple[str, str], list[ReviewFinding]]:
    """Prefer a shared description; allow a shared title to cover a rewritten description.

    Keep complete fields and their case. Matching fragments can join unrelated
    failures that mention the same operation. A title cannot bridge two groups
    that already have independent descriptions corroborated by different titles.
    """
    titles_by_description: dict[str, set[str]] = defaultdict(set)
    descriptions_by_title: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        title, description = finding.title.strip(), finding.description.strip()
        titles_by_description[description].add(title)
        descriptions_by_title[title].add(description)

    shared_descriptions = {
        description
        for description, titles in titles_by_description.items()
        if description and len(titles) > 1
    }
    rewrite_labels: dict[str, tuple[str, str]] = {}
    for title, descriptions in descriptions_by_title.items():
        if len(descriptions) == 1 and "" not in descriptions:
            continue
        shared = descriptions & shared_descriptions
        if len(shared) == 1:
            # The original connects a title rewrite and a description rewrite.
            (original_description,) = shared
            rewrite_labels[title] = ("description", original_description)
        else:
            rewrite_labels[title] = ("title", title)

    groups: dict[tuple[str, str], list[ReviewFinding]] = defaultdict(list)
    for finding in findings:
        title, description = finding.title.strip(), finding.description.strip()
        label = ("description", description)
        if title and description not in shared_descriptions:
            label = rewrite_labels.get(title, label)
        groups[label].append(finding)
    return groups


def distinguish_judged_claims(
    findings: list[ReviewFinding], prior_findings: Sequence[ReviewFinding]
) -> tuple[ReviewFinding, ...]:
    """Retain independent claims at a shared anchor without changing proposal provenance.

    An unambiguous location preserves its historical identity. For a collision,
    reuse an earlier matching identity where possible. Structural proof keeps
    the existing tier-based merge. All admission occurrences remain available
    for candidate receipts before the round deduplicates or caps them.
    """
    by_location: dict[str, list[ReviewFinding]] = defaultdict(list)
    for finding in findings:
        by_location[finding.content_hash].append(finding)
    prior_by_location: dict[str, list[ReviewFinding]] = defaultdict(list)
    for previous in prior_findings:
        location = previous.location_hash
        if location in by_location:
            prior_by_location[location].append(previous)

    replacements = {}
    for location, current in by_location.items():
        prior = prior_by_location[location]
        if not prior and len(current) == 1:
            continue
        if any(f.evidence_tier is not EvidenceTier.JUDGED for f in (*current, *prior)):
            continue
        groups = _claim_groups(current)
        if len(groups) == 1 and all(f.claim_key is None for f in prior):
            continue
        matches = {}
        for label, members in groups.items():
            titles = {m.title.strip() for m in members} - {""}
            descriptions = {m.description.strip() for m in members} - {""}
            matches[label] = {
                f.claim_key
                for f in prior
                if f.title.strip() in titles or f.description.strip() in descriptions
            }
        # A previous title and description can point to different current claims.
        # Reuse its identity only when the match is unique in both directions.
        references = Counter(key for keys in matches.values() for key in keys)
        for (field, text), members in groups.items():
            prior_keys = matches[field, text]
            if len(prior_keys) == 1 and references[next(iter(prior_keys))] == 1:
                claim_key = next(iter(prior_keys))
                if claim_key is None:
                    continue
            else:
                claim_key = hashlib.sha256(f"{field}\0{text}".encode()).hexdigest()
            for member in members:
                content_hash = compute_finding_content_hash(
                    member.file_path,
                    line_start=member.line_start,
                    line_end=member.line_end,
                    finding_type=member.finding_type,
                    source_side=member.source_side,
                    claim_key=claim_key,
                )
                replacements[member.finding_id] = member.model_copy(
                    update={"claim_key": claim_key, "content_hash": content_hash}
                )
    return tuple(replacements.get(f.finding_id, f) for f in findings)
