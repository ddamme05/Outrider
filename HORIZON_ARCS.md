# HORIZON ARCS — Multi-Provider and Multi-Model Review

**Status:** Draft architecture report for discussion and later adoption into the Outrider repository  
**Prepared:** 2026-07-19  
**Scope:** OpenAI GPT-5.6 compatibility, Claude + GPT routing, independent multi-model review, refusal safety, cost control, and staged delivery  
**Authority:** Advisory. This document does not amend `DECISIONS.md`, `docs/spec.md`, or an approved feature spec. Accepted changes must follow the repository's normal spec and decision process.

---

## Executive recommendation

Outrider should pursue a Claude + GPT architecture, but it should not make GPT a drop-in whole-pipeline replacement and it should not make either model the sole judge of the other model's work.

The recommended target is:

1. Keep the existing `OpenAICompatibleProvider` + `HostProfile` architecture. A separate `OpenAIProvider` is not needed.
2. Do not production-admit the GPT-5.6 host under the current `json_object` contract. A silent deflection is observationally identical to a legitimate zero-finding review.
3. Add provider routing by **operation**, not merely by graph node. Some logical nodes make more than one materially different LLM call.
4. Build a proof-preserving strict structured-output contract for GPT Analyze. This is the feature that can make GPT a safe production analyzer.
5. Once that contract passes paid wire and proof-boundary gates, run Claude and GPT as **independent parallel analyzers** over selected files.
6. Reconcile their findings through deterministic proof validation, policy-owned severity, union/deduplication, and human adjudication—not through an LLM judge.
7. Start selectively: Claude remains the normal reviewer, while GPT-5.6 Sol independently reviews at most three high-risk DEEP files per review. Expand only when validated marginal findings per dollar justify it.

The central distinction is:

> Routing decides **where** a call goes. A response contract decides **whether that provider can safely serve the operation**. An ensemble policy decides **how independent outputs are combined**.

These are complementary features, not substitutes.

---

## 1. Why this horizon exists

The GPT-5.6 integration established that Outrider's provider abstraction is broadly correct. The OpenAI SDK path, host profiles, host-qualified pricing, event identity, caching, and scorecard seams can accommodate a native OpenAI host.

The admission work also exposed a deeper compatibility boundary: a provider may be transport-compatible while still failing the semantic contract required by a security reviewer.

In production `json_object` mode, the following response can mean either:

- the model completed the review and found no vulnerabilities; or
- the model silently deflected/refused the task while remaining inside the requested JSON shape.

```json
{
  "content": "{\"findings\":[]}",
  "refusal": null,
  "finish_reason": "stop"
}
```

Those outcomes are byte-for-byte indistinguishable at the provider boundary. No parser, retry policy, event consumer, or replay process can reconstruct information that the API did not transmit.

The result is not "OpenAI cannot be integrated." It is:

> GPT-5.6 cannot safely serve every current Outrider operation through the current soft-output contract.

That distinction creates a useful roadmap: improve routing so providers can be adopted incrementally, and improve the Analyze response contract so GPT can eventually become an independent reviewer.

---

## 2. Current architecture, stated accurately

### 2.1 Provider architecture

The accepted architecture is one `OpenAICompatibleProvider` parameterized by validated host data in a frozen `HostProfile`. Native OpenAI remains a host profile, not a separate provider class.

The provider boundary should be extended when a host has a novel wire behavior. It should not be duplicated merely because the hostname is `api.openai.com`.

### 2.2 Current host selection

Today Outrider selects one host for the whole review. `OUTRIDER_LLM_HOST` resolves at the composition root and the resulting provider is shared by the LLM-using operations.

This means the current application cannot safely express:

```text
Analyze    → Claude
Trace      → GPT
Synthesize → Claude
```

without introducing a routing layer and changing review-wide host-identity assumptions.

### 2.3 Logical nodes

Outrider's seven logical nodes are:

```text
intake → triage → analyze → trace → synthesize → hitl → publish
```

`analyze_file` and `analyze_aggregate` are physical graph vertices under the logical Analyze identity.

### 2.4 Patch is not a node

Patch generation is an optional sub-pass inside Synthesize. It receives already-admitted findings and attempts to produce exact replacement lines for eligible HIGH/CRITICAL single-line findings.

It does not:

- discover findings;
- decide whether a finding is valid;
- set severity;
- remove findings;
- perform a second code review.

Patch generation uses a separately configured `patch_model`, but its LLM event currently carries `node_id="synthesize"`. Therefore, a router keyed only by `node_id` cannot send Synthesize summary generation and patch generation to different providers.

This is why the future routing key must include an operation or call-purpose identity.

---

## 3. What the paid GPT-5.6 evidence established

The captured work established several positive facts:

- GPT-5.6 accepts the production Chat Completions route when the host-specific token-limit parameter is shaped correctly.
- `reasoning_effort="none"` is accepted for the tested route.
- The expected default service-tier echo was observed on successful rows.
- Cold and warm cache behavior was observed, including separately reported cache writes and reads.
- The trace-ranking scenario produced a valid permutation and ranked the expected candidate first.
- The bounded patch scenario produced the intended anchored single-line replacement.

It also established two negative facts that must not be softened:

1. `{"findings":[]}` is not a refusal discriminator. It is also the valid representation of a clean review.
2. HTTP 401 is not refusal behavior. It maps to a terminal authentication error and has no completed assistant response to normalize.

The refusal-discovery runs produced completed GPT-5.6 responses with `finish_reason="stop"`, `message.refusal=null`, and an empty findings array. Sol produced clean observations across the discovery prompts. Luna also produced completed negative observations, with at least one separate 401 arm remaining an authentication failure rather than refusal evidence.

The important conclusion does not depend on guessing why the model returned empty findings:

> The current wire response contains no reliable signal that distinguishes a completed clean review from a silent deflection.

Additional repetitions cannot resolve a missing discriminator. They can estimate behavior frequency, but they cannot make identical responses distinguishable.

---

## 4. Why the easy fixes are unsafe

### 4.1 Accepting empty findings as refusal

Rejected. It would convert silent refusals into false-negative clean reviews and erase the exact guarantee the admission gate is intended to protect.

### 4.2 Treating 401 or message text as a policy refusal

Rejected. A 401 is an authentication error. Any future specialized policy classification must rely on a documented, stable machine-readable discriminator—not status-code overloading or message substring matching.

### 4.3 Refusal phrase detection

Rejected. It is brittle, language-dependent, and cannot detect schema-conformant empty JSON.

### 4.4 Adding a soft `status="completed"` field

Insufficient as the load-bearing guarantee. In `json_object` mode this remains a model-authored assertion; a deflecting model can emit the attestation and an empty findings array.

### 4.5 Failing every empty review

Rejected. It would make legitimate clean reviews unusable and incentivize false-positive filler findings.

### 4.6 Moderation as a refusal detector

Insufficient. Moderation can classify input or output risk; it does not establish whether the review model completed the requested code analysis. Security-review inputs also legitimately contain vulnerability language.

### 4.7 A second verifier call

Not an equivalent discriminator. A second model can also refuse, miss the issue, or be anchored by the first result. It adds spend without restoring the missing wire fact.

### 4.8 A separate native provider class

Rejected. A new class would rearrange code but could not extract information absent from the API response. The existing compatible-provider boundary remains the correct extension point.

### 4.9 Mechanical required-completion

Rejected. The Fireworks experiment showed that mechanically forcing optional proof fields to be present caused fabricated `query_match_id` and `trace_path` metadata on a JUDGED finding. This crosses Outrider's proof boundary.

### 4.10 Switching endpoints without redesigning the contract

Insufficient. Responses may be a useful future surface, but an endpoint migration alone does not make Outrider's proof schema safe under strict structured output.

---

## 5. The enabling feature: proof-preserving strict structured output

OpenAI Structured Outputs provide a programmatically detectable refusal variant, but OpenAI's supported strict subset requires every field to be `required` and objects to reject additional properties.

The solution is not to mark every property in the current flat schema required. The solution is to make the schema express Outrider's semantic branches.

Conceptually, and `EvidenceTier` is exactly `{OBSERVED, INFERRED, JUDGED}` (`policy/findings.py`, `evidence-tier-schema-enforced`) — there is no "other tiers" bucket; all three must be explicit branches:

```text
Finding
├── OBSERVED
│   ├── required observed fields
│   └── query_match_id is required; trace_path is not a permitted property
├── INFERRED
│   ├── required inferred fields
│   └── trace_path is required and NON-EMPTY; query_match_id is not permitted
└── JUDGED
    ├── required judged fields
    └── neither query_match_id nor trace_path is a permitted property
```

Every property inside a branch can be required while irrelevant proof fields are structurally absent. This avoids both failure modes:

- the schema does not force the model to invent inapplicable proof metadata;
- the API can surface refusal separately from a successful empty result.

### 5.1 Proposed implementation shape

1. Define a strict OpenAI wire DTO/schema distinct from the canonical internal response model.
2. Keep the root schema an object; place evidence-tier alternatives below the root.
3. Make every branch closed with `additionalProperties:false`.
4. Include only semantically valid proof fields in each branch.
5. Send the schema through `json_schema` with `strict:true`.
6. Handle the API refusal variant before attempting content parsing.
7. Translate a validated wire DTO into the existing canonical `AnalyzeResponseRaw` representation.
8. Retain every existing post-parse proof check. Strict syntax does not replace registry resolution, source matching, severity policy, or finding admission.
9. Treat this as a provider-boundary/shaper extension and rotate all required contract versions and profile digests.

### 5.2 Required paid evidence

The feature does not ship merely because the schema compiles. A paid fixture must prove:

- a legitimate zero-finding result is a successful content response;
- an elicited refusal appears in the API-owned refusal channel;
- **all three proof tiers compile and parse under strict**;
- JUDGED findings cannot carry OBSERVED-only (`query_match_id`) or INFERRED-only (`trace_path`) proof metadata;
- OBSERVED findings cannot omit their required `query_match_id`;
- **INFERRED findings carry a required non-empty `trace_path`; a missing, empty, or malformed `trace_path` fails closed** (`evidence-tier-schema-enforced`);
- no required branch encourages fabricated proof metadata;
- usage, caching, tier echo, and error translation remain correctly accounted;
- the production parser and persisted event carry the same interpretation.

### 5.3 Scope across operations

Analyze is the known blocker because an empty findings array is a valid successful result. Before whole-pipeline admission, every other GPT-served operation must also be checked for a valid empty/no-op result that could conceal refusal.

The standard is operation-specific:

- a trace ranking that must return a complete permutation naturally fails closed on empty output;
- patch generation may legitimately return no suggestion, so its refusal/no-op distinction deserves explicit review;
- summaries and triage outputs must be checked against their own valid-empty semantics.

Strict Analyze admission is necessary, but whole-host admission requires the complete operation matrix.

---

## 6. The routing feature: route by operation, not only node

### 6.1 Why node-only routing is too coarse

One logical node can own multiple calls with different purposes:

- Synthesize performs patch generation and summary generation.
- Analyze has DEEP/STANDARD model tiers and multiple physical graph vertices under one logical identity.
- Future verification or critique calls would be distinct from primary generation even if they occur in the same node.

The route identity should therefore look conceptually like:

```text
(node_id, operation)
```

Possible operation names:

```text
triage.classify
analyze.primary
analyze.secondary
trace.rank
synthesize.patch_generation
synthesize.summary
```

Names are illustrative. The approved spec should freeze the vocabulary and ownership.

### 6.2 Routing authority

Routing belongs at the composition boundary. The graph or an injected routing provider should receive a validated route plan; individual node bodies must not reread environment variables or select vendors ad hoc.

A route configuration SELECTS a candidate tuple:

- provider/profile identity;
- model slug;
- reasoning posture;
- output/shaper contract;
- operation;
- per-operation budget;
- fallback policy.

**Admission is NOT a route-configured field.** A closed, versioned capability registry derives whether an admitted tuple covers the exact route. The key must include everything that changes the wire evidence — reasoning posture and the profile-contract digest, not just the host:

```text
(
  profile_id,
  profile_contract_digest,
  model,
  operation,
  output_contract,
  reasoning_posture,
)
```

so evidence earned for reasoning-off (or an older profile digest) cannot admit a reasoning-on or contract-changed route. Each registry record binds its wire fixture, scorecard/admission evidence, and contract version. Composition **fails before constructing any provider** if the route's exact key is unadmitted; Arc 1 must test that a reasoning or digest mismatch fails there. Operators must never be able to declare admission through configuration — the same principle as the no-ship composition-root refusal (Arc 0). Routing decides *where*; the registry decides *whether it may serve*.

### 6.3 Audit and replay requirements

Mixed-provider reviews invalidate the assumption that one review-wide host triad describes every completion.

The implementation must ensure:

- each LLM call records the actual profile, model, reasoning setting, and profile-contract digest;
- node-completion events derive identity from the calls or route plan that actually served the operation;
- cache keys include the actual provider/model/shaper route;
- historical event replay never infers host from a model slug;
- review-level metrics can report the set of providers that received content;
- idempotent re-emission remains stable across routing changes.

### 6.4 Provider lifecycle

The composition root owns all provider instances. A routing layer delegates calls but does not hide resource ownership.

Required behavior:

- construction failure closes providers already acquired;
- normal shutdown attempts every provider close;
- one provider's close failure does not leak another provider;
- provider instances are reused across calls so their connection pools and cache behavior remain effective.

### 6.5 Fallback is a separate policy

Routing must not silently imply fallback.

In particular:

- a safety refusal must not automatically be sent to another provider as a way to bypass the first provider's safety decision;
- an authentication or malformed-wire error should fail according to its taxonomy, not trigger unbounded paid retries;
- any quality fallback must be explicitly budgeted, audited, bounded to one transition, and independently admission-tested.

---

## 7. The Claude + GPT review model

### 7.1 Evidence and nuance

Research on LLM-as-a-judge and self-refinement has found self-preference: models can score or preserve their own generations more favorably than equally good outputs from other sources. More recent work also warns that raw preference measurements can conflate bias with real quality differences. The correct design response is not to assume every same-family judgment is corrupt; it is to avoid making ungrounded model judgment the final authority.

Relevant research:

- Panickssery, Bowman, and Feng, *LLM Evaluators Recognize and Favor Their Own Generations* (2024): https://arxiv.org/abs/2404.13076
- Xu et al., *Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement* (ACL 2024): https://aclanthology.org/2024.acl-long.826/
- Chen et al., *Do LLM Evaluators Prefer Themselves for a Reason?* (2025): https://arxiv.org/abs/2504.03846
- Chen et al., *Beyond the Surface: Measuring Self-Preference in LLM Judgments* (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.86/

These studies motivate diversity and objective grading. They do not by themselves prove a specific Claude-versus-GPT bias magnitude for Outrider's code-review workload. That magnitude must be measured on Outrider's own fixtures.

### 7.2 Recommended topology

```text
                      identical canonical code/context
                         ┌──────────┴──────────┐
                         │                     │
                  Claude Analyze        GPT Analyze
                   independently         independently
                         │                     │
                         └──────────┬──────────┘
                                    │
                       deterministic proof validation
                                    │
                          union + dedup + provenance
                                    │
                    unresolved consequential conflicts
                                    │
                                  HITL
                                    │
                               Synthesize
```

### 7.3 Independence rules

Before generation:

- both analyzers receive semantically equivalent canonical context;
- neither analyzer sees the other model's findings;
- neither analyzer sees provider/model provenance;
- selection for the secondary lane is deterministic and happens before primary findings exist.

The secondary reviewer must not run only when Claude says "clean" or only when Claude reports a vulnerability. Both strategies anchor the second model on the first model's behavior and preserve primary blind spots.

### 7.4 Union, not intersection

Requiring both models to agree would discard the unique findings that justify model diversity.

The merge policy should be:

- either model may propose a finding;
- deterministic proof checks decide whether its evidence is admissible;
- policy, not a model, sets severity;
- duplicates merge;
- agreement is recorded ONLY as ensemble provenance + metrics — it must **never** alter `evidence_tier`, `ReviewFinding.confidence` (deterministic `@computed_field`: OBSERVED=0.9 / INFERRED=0.75 / JUDGED=0.5 per `confidence-is-computed-not-assigned`), severity, or publication eligibility;
- disagreement does not automatically delete a finding;
- unresolved high-impact disagreement routes to HITL.

### 7.5 Provenance

The system should internally retain whether a finding was proposed by:

- Claude only;
- GPT only;
- both independently;
- a deterministic query;
- a traced/verified evidence path.

That provenance belongs in audit and metrics. It should normally be hidden from any later model asked to inspect finding content, since source identity can induce preference and models can sometimes infer family from style even after labels are removed.

### 7.6 Synthesize is not the judge

Synthesize may summarize and present the accepted finding union. It must not silently remove findings because its model disagrees.

The canonical finding collection must remain application-owned and travel separately from generated prose. Summary generation is presentation; it is not adjudication.

### 7.7 Patch generation

Patch generation is bounded transformation rather than finding adjudication. Cross-family patch generation may be useful:

- GPT can propose a constrained patch for a Claude-originated finding;
- Claude can propose one for a GPT-originated finding;
- deterministic anchoring and safety checks accept or discard the replacement;
- failure leaves the finding intact and unpatched.

This is optional optimization, not a prerequisite for the dual-review architecture.

### 7.8 If an LLM critique stage is later added

It should not become the final arbiter. At minimum:

- findings are source-blinded and order-randomized;
- Claude critiques GPT-originated claims and GPT critiques Claude-originated claims symmetrically;
- both critique outcomes remain advisory;
- deterministic proof or HITL owns the final consequential decision;
- the stage has its own budget and admission metrics.

Independent generation plus objective reconciliation is preferable to an author/critic chain.

---

## 8. Cost model

### 8.1 Current standard rates

Prices below are per million tokens and were verified on 2026-07-19. They are time-sensitive and must be refreshed before implementation or a pricing-version bump.

| Model | Input | Cached input | Cache write (Claude: 5m) | Output |
|---|---:|---:|---:|---:|
| Claude Sonnet 5, introductory through 2026-08-31 | $2.00 | $0.20 | $2.50 | $10.00 |
| Claude Sonnet 5, from 2026-09-01 | $3.00 | $0.30 | $3.75 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $0.10 | $1.25 | $5.00 |
| GPT-5.6 Sol | $5.00 | $0.50 | $6.25 | $30.00 |
| GPT-5.6 Terra | $2.50 | $0.25 | $3.125 | $15.00 |
| GPT-5.6 Luna | $1.00 | $0.10 | $1.25 | $6.00 |

Sources:

- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
- Claude API pricing: https://platform.claude.com/docs/en/about-claude/pricing

OpenAI Sol and Terra long-context requests above the documented threshold use higher full-request rates. The current Outrider design intends to prevent normal admission into that tier; this horizon's examples use short-context pricing.

### 8.2 Named modeling assumptions

To compare architectures, this document reuses the existing Outrider 30-file planning model:

- 10 DEEP files;
- 15 STANDARD files;
- 5 SKIM/SKIP files with no Analyze call;
- per DEEP call: 3k reusable prefix, 5k uncached input, 2k output;
- per STANDARD call: 2k input, 500 output for the simple comparison;
- reusable-prefix average: 90% reads, 10% writes across ten DEEP calls;
- approximately $0.02 for non-Analyze LLM overhead.

These are planning assumptions, not production measurements. Tokenizers differ by provider and model; output length and cache-hit rates must be measured from real events.

### 8.3 Approximate per-call costs

Under those assumptions:

| DEEP reviewer | Approx. cost per file |
|---|---:|
| Claude Sonnet 5, introductory | $0.031 |
| Claude Sonnet 5, standard | $0.047 |
| GPT-5.6 Sol | $0.088 |
| GPT-5.6 Terra | $0.044 |
| GPT-5.6 Luna | $0.018 |

The OpenAI costs include the modeled mix of cache reads/writes. Output remains the dominant cost for Sol; prompt caching does not discount output tokens.

### 8.4 Approximate 30-file review costs

This `$0.40` is the **optimized cached introductory baseline** (90%-cache-read prefix, intro Sonnet 5 pricing) — NOT `COST_ANALYSIS.md`'s naive or standard-price figures; do not conflate them when quoting a multiplier.

| Strategy | Added secondary cost | Total during Sonnet introductory pricing | Approx. multiplier |
|---|---:|---:|---:|
| Claude-only (optimized cached, intro) | — | $0.40 | 1.0× |
| Sol on all DEEP + Luna on STANDARD | +$0.96 | $1.36 | 3.4× |
| Terra on all DEEP + Luna on STANDARD | +$0.52 | $0.92 | 2.3× |
| Luna on every analyzed file | +$0.25 | $0.65 | 1.6× |
| Sol on only three highest-risk DEEP files | +$0.26 | $0.67 | 1.7× |

After Claude Sonnet 5's introductory period, the modeled Claude-only baseline returns to approximately $0.56. The secondary OpenAI increment is unchanged under the quoted OpenAI rates, so the modeled totals become approximately:

- full Sol/Luna secondary lane: $1.52;
- full Terra/Luna lane: $1.08;
- Luna full lane: $0.81;
- three-file Sol selective lane: $0.82.

### 8.5 Monthly illustration

At 100 representative reviews/day for 30 days:

| Strategy | Approx. monthly model spend during Sonnet introductory pricing |
|---|---:|
| Claude-only | $1,200 |
| Three-file selective Sol | $2,000 |
| Full Terra/Luna secondary lane | $2,750 |
| Full Sol/Luna quality-first lane | $4,070 |

These figures exclude infrastructure, storage, support, retries, data-residency uplifts, taxes, and negotiated discounts.

### 8.6 Recommended initial budget posture

Do not run both flagship models over every file.

Initial policy:

```text
All analyzed files:
  existing Claude route

Highest-risk DEEP files:
  independent GPT-5.6 Sol lane

Secondary cap:
  min(3 files, ceil(25% of DEEP files)),
  plus a hard secondary dollar ceiling
```

The cost illustration uses three secondary files. On reviews where the percentage cap selects fewer files, actual spend is lower. This targets an approximate $0.65–$0.70 representative review during the introductory Claude pricing period rather than $1.30–$1.50.

### 8.7 Required cost controls

Mixed-provider routing needs more than the existing review-wide token budget because equal token counts have different dollar values across hosts.

Required controls:

1. **Primary and secondary budgets.** Exhausting the secondary allocation disables only the extra reviewer; it does not degrade the primary review.
2. **Host-qualified preflight estimates.** Use the selected `(profile_id, model)`, rendered prompt bound, expected output ceiling, cache posture, service tier, and long-context policy.
3. **File and call caps.** Bound secondary files and exactly how many paid transitions one operation may trigger.
4. **No unbudgeted cross-provider retries.** A failed GPT call must not silently create Claude and GPT retry chains.
5. **Actual-cost reconciliation.** Persist actual per-call cost and compare it with the preflight estimate.
6. **Marginal-value measurement.** Record validated unique findings and accepted proof improvements per secondary dollar.
7. **Per-route dashboards.** Review totals must expose primary cost, secondary cost, unpriced calls, and completeness.

### 8.8 Cost expansion criterion

The secondary lane expands only if measurements show a worthwhile marginal return. Candidate metrics:

- unique admitted findings per 100 secondary calls;
- unique CRITICAL/HIGH findings per $100;
- false-positive rate of GPT-only findings;
- percentage of GPT-only findings confirmed by deterministic proof or HITL;
- overlap rate with Claude;
- incremental p50/p95 wall-clock latency;
- cache-write/read realization;
- cost per completed review by route policy.

A model being impressive in examples is not enough. The union must produce measurable validated recall that clears its cost and operational burden.

### 8.9 Human-disagreement cost (not just model execution)

§8.1–§8.6 price only LLM execution. But the design routes consequential unresolved conflicts to HITL, and dual review *increases* disagreement volume — a real cost the model-token budget hides. Measure and budget:

- cross-model disagreement rate (per review, per file class);
- HITL minutes per adjudicated case + loaded human cost;
- HITL queue depth and p50/p95 wait;
- percentage of disagreements requiring escalation.

These belong in Arc 3's expansion/stop criterion alongside the per-dollar recall metrics: a secondary lane that adds validated findings but floods HITL may still fail its exit gate.

---

## 9. Latency, privacy, and operational effects

### 9.1 Latency

Independent lanes should execute concurrently after deterministic routing and context construction.

Parallel execution means spend adds, but wall-clock time trends toward the slower lane plus merge overhead rather than the sum of both lanes. Tail latency can still worsen because the review waits for two providers and inherits the slower provider's p95 behavior.

The secondary lane should have a bounded timeout. Timeout behavior must be explicit: normally ship the completed primary review with a recorded secondary degradation rather than retrying without limit.

### 9.2 Privacy and egress

A mixed review sends source code to more than one vendor. This is a product and trust-boundary change, not merely internal routing.

The review record should expose:

- which providers received content;
- which operations each provider served;
- the privacy profile and verification date for each provider;
- whether any configured retention/ZDR posture differed;
- whether a customer policy prohibited multi-provider egress.

Customers should be able to select a Claude-only policy even if the mixed route is generally available.

### 9.3 Rate limits and concurrency

Load is distributed across vendors, which may improve aggregate throughput, but each provider has independent request/token limits. The router needs per-provider concurrency and backpressure rather than one global semaphore.

### 9.4 Caching

Providers do not share prompt caches. Each route maintains its own cache identity and pays its own cold write. Stable prefixes must remain stable separately for Claude and GPT. Small reviews may not realize enough warm calls to amortize both caches.

### 9.5 Audit volume

Full dual Analyze roughly doubles Analyze LLM-call events, token accounting, and response artifacts. The audit model must preserve source provenance without making dashboards imply that agreement count is equivalent to truth.

---

## 10. Horizon arcs

The arcs below are ordered by dependency. Arc 1 and Arc 2 can be designed in parallel, but production dual review requires both.

### Arc 0 — Close the current native-host admission honestly

**Objective:** Preserve the implementation and evidence without weakening the refusal contract.

**Decision:** GPT-5.6 is implemented/evaluable but not production-admitted as a whole-pipeline host under the current contract.

**Actions:**

- record the indistinguishable-empty-result blocker in the spec's Actual Outcome;
- record 401 observations as authentication errors, not refusals;
- do not accept empty findings as a refusal predicate;
- do not continue paid discovery merely to repeat the same ambiguity;
- keep Anthropic as the production default;
- ensure a `WIRE-PENDING` label cannot become the only barrier to production selection.

**Merge posture:** The cleanest posture is to hold the production-selectable host until admission. If groundwork must merge, composition must hard-refuse production OpenAI selection while preserving an explicitly scoped probe/eval construction path.

**Exit rule:** Current arc is closed as not admitted, with no production configuration capable of silently enabling the host.

### Arc 1 — Operation identity and provider routing

**Objective:** Allow one review to use different admitted providers/models for different operations.

**Core changes:**

- define a closed operation identity separate from `node_id`;
- define one authoritative route plan at the composition root;
- route each operation to a provider/profile/model/reasoning/output contract;
- update cache, completion-event identity, cost aggregation, and privacy reporting for mixed routes;
- preserve deterministic provider lifecycle ownership;
- define fallback separately from routing.

**Initial validation:** Configure multiple routes that still point to Anthropic models and prove byte/event equivalence before introducing a second production provider.

**Non-goal:** Dual Analyze. This arc provides routing infrastructure only.

**Exit rule:** A review can execute two operation routes backed by different injected provider doubles with correct cost, audit identity, cache keys, shutdown, and historical replay behavior.

### Arc 2 — Proof-preserving strict GPT Analyze

**Objective:** Make GPT Analyze refusal distinguishable from a legitimate zero-finding result without weakening proof metadata.

**Feasibility gate — DO THIS FIRST, before any plumbing.** OpenAI's docs confirm the design is *plausible* (strict requires a root object, permits nested `anyOf`, requires all fields, exposes refusals separately) — but docs don't prove the exact three-tier Outrider schema compiles or that GPT-5.6 refuses in the API channel. Generate the complete strict schema and run ONE bounded paid capture proving: (a) the schema is accepted; (b) a legitimate empty result is a content response; (c) an elicited refusal uses the separate wire refusal field; (d) all three proof branches (OBSERVED/INFERRED/JUDGED) compile and parse. **Stop Arc 2 immediately if schema acceptance OR refusal discrimination fails** — everything below is wasted otherwise.

**Core changes (only after the feasibility gate passes):**

- strict evidence-tier-discriminated wire schema (all three tiers explicit);
- wire DTO → canonical response translation;
- API-owned refusal handling before parse;
- shaper/profile contract rotation;
- proof-boundary unit tests and paid wire fixtures;
- model-specific scorecard and prompt-transfer evaluation.

**Non-goal:** Mechanical required-completion of the existing flat schema.

**Exit rule:** Paid fixtures demonstrate distinct clean-empty and refusal outcomes, and all proof-boundary/schema/eval gates pass for every GPT model proposed for Analyze.

### Arc 3 — Selective independent dual Analyze

**Objective:** Gain cross-family recall while controlling cost and avoiding model-as-final-judge bias.

**Initial route:**

- Claude retains the primary lane;
- GPT-5.6 Sol independently reviews at most three deterministically selected high-risk DEEP files;
- both receive equivalent canonical context;
- neither sees the other's output before generation.

**Merge:** Deterministic proof validation → union → dedup → provenance → HITL for consequential unresolved conflicts.

**Privacy prerequisite (blocks Arc 3 exit; a second vendor receives source code here, which `#013`/`#015`/`#066` currently frame as the *singular* configured provider):**

- multi-vendor egress defaults OFF;
- explicit operator/customer opt-in required before any second vendor receives content;
- the complete provider + retention/ZDR set is disclosed before execution;
- a customer policy can prohibit multi-provider egress and force Claude-only;
- the actual recipients per review remain audit-visible.

This is a decision that must land (spec + `DECISIONS.md`) BEFORE the arc exits, not after — dual egress is a product/trust-boundary change, not internal routing.

**Required metrics:** Claude-only, GPT-only, union, intersection/agreement, marginal validated recall, false positives, cost, latency, AND the human-disagreement costs in §8.9.

**Exit rule:** Exit only when ALL of these hold: the selective lane shows an approved improvement in validated recall per dollar; the secondary cost ceiling is not exceeded; the precision/proof gates are not weakened; the human-load gate passes; **and** the accepted multi-vendor privacy decision has landed.

### Arc 4 — Cost-aware expansion and tier experiments

**Objective:** Determine where cheaper GPT family members preserve the value of the independent lane.

**Experiments:**

- Terra versus Sol on selected DEEP files;
- Luna on STANDARD files only if its node-specific quality and refusal contracts pass;
- varying secondary caps by risk class;
- cache realization and output-token reduction;
- optional async/service-tier experiments only under separately versioned pricing and latency evidence.

**Expansion rule:** Change one dimension at a time. Do not combine model downgrade, prompt rewrite, reasoning change, and routing expansion in one evaluation.

**Exit rule:** A stable policy maps risk/tier to the cheapest model that clears the quality floor, with measured costs replacing the planning estimates in this document.

### Arc 5 — Optional cross-family transformation and critique

**Objective:** Explore cross-family patch generation or advisory critique after the independent ensemble is stable.

**Constraints:**

- patch generation remains bounded and deterministic-gated;
- critique is source-blinded, symmetric, advisory, and separately budgeted;
- no LLM judge becomes severity or final-admission authority;
- no refusal is automatically bypassed through another provider.

**Exit rule:** Measured benefit beyond independent generation and deterministic reconciliation. If there is no incremental value, do not ship the complexity.

---

## 11. Evaluation plan

### 11.1 Do not use one model's opinion as ground truth

Evaluation should use:

- curated expected findings;
- deterministic structural graders;
- proof-boundary validation;
- human adjudication for semantic quality;
- model-family ablations.

### 11.2 Required comparison columns

For every admitted dual-review workload:

| Column | Purpose |
|---|---|
| Claude alone | Existing baseline |
| GPT alone | Independent capability and failure modes |
| Claude ∪ GPT | Combined recall and precision |
| Claude ∩ GPT | Agreement signal only |

The intersection must never become the shipping filter by default; it would sacrifice unique valid findings.

### 11.3 Provenance-safe grading

Human adjudicators may need provenance for debugging, but the initial verdict should be blind where practical. Automated graders should operate on normalized finding/evidence fields, not model writing style.

### 11.4 Required adversarial scenarios

- legitimate zero-finding review;
- API refusal;
- soft in-band deflection;
- malformed structured output;
- JUDGED finding attempting to carry OBSERVED (`query_match_id`) or INFERRED (`trace_path`) proof;
- OBSERVED finding missing or fabricating its query identity;
- INFERRED finding with a missing, empty, or malformed `trace_path` (must fail closed);
- INFERRED finding with a valid non-empty `trace_path` (positive control);
- Claude-only true finding;
- GPT-only true finding;
- semantically duplicate findings with different prose/spans;
- conflicting severity claims;
- one provider timeout or auth failure;
- exhausted secondary budget;
- cache cold/warm transition for each provider;
- historical replay across a route-plan/version change.

---

## 12. Guardrails and non-goals

The horizon does not authorize:

- a separate native `OpenAIProvider` merely for branding;
- accepting empty findings as refusal;
- message-text or HTTP-status refusal heuristics;
- mechanical required-completion of optional proof metadata;
- per-call environment-variable routing inside nodes;
- unbounded retries or silent cross-provider fallback;
- model-owned severity;
- ensemble agreement altering `evidence_tier`, `ReviewFinding.confidence`, severity, or publication;
- admission declared through route/operator configuration rather than a versioned capability registry;
- consensus-only finding publication;
- an LLM judge as final arbiter;
- sending code to a second vendor without customer/privacy policy accounting;
- full dual review before selective marginal value is measured;
- changing endpoint, reasoning, prompt, schema, and model tier simultaneously.

---

## 13. Decision matrix

| Question | Recommended answer |
|---|---|
| Ship GPT-5.6 as the current whole-pipeline host? | No, not under the current soft refusal contract |
| Discard the existing integration? | No; retain it as groundwork and evidence |
| Build a separate OpenAI provider? | No |
| Add routing? | Yes, by operation/call purpose |
| Let GPT judge Claude or Claude judge GPT? | Not as final authority |
| How should findings combine? | Deterministic validation + union + dedup + provenance |
| What resolves the Analyze blocker? | Proof-preserving strict structured output |
| Start with dual review on every file? | No |
| Initial GPT scope | At most three high-risk DEEP files per review |
| Initial GPT model | Sol for the quality baseline; evaluate Terra later |
| Use Luna immediately for STANDARD? | Only after operation-specific admission and quality evidence |
| Automatic fallback after refusal? | No |
| Expansion criterion | Validated marginal findings per dollar and acceptable latency/precision |

---

## 14. Recommended sequence

```text
Now
 └─ Close native GPT host as implemented but not production-admitted

Next, in parallel
 ├─ Arc 1: operation identity + provider routing
 └─ Arc 2: proof-preserving strict GPT Analyze contract

Then
 └─ Arc 3: selective independent Claude + GPT Analyze

Only after measurement
 ├─ Arc 4: Terra/Luna and broader cost-aware expansion
 └─ Arc 5: optional cross-family patch/critique experiments
```

This sequence avoids two common mistakes:

1. weakening the safety contract simply to make the current host pass; and
2. paying for a full ensemble before measuring whether the second reviewer produces validated marginal value.

---

## 15. Open questions for the approving spec

1. What exact closed vocabulary identifies LLM operations?
2. Does the router inject one provider per operation or expose a protocol-compatible delegating provider?
3. How do Analyze completion events represent two providers in one logical node?
4. What normalized finding key deduplicates cross-model descriptions without merging distinct vulnerabilities?
5. What deterministic risk features select the secondary files before either model runs?
6. What is the initial secondary dollar ceiling per review and per customer?
7. Which customer/privacy configurations forbid multi-provider egress?
8. What constitutes enough marginal recall to expand beyond three DEEP files?
9. Does any operation other than Analyze have a valid empty/no-op outcome that can mask refusal?
10. Is the proof-preserving strict schema host-specific wire data or a shared canonical schema version?
11. If OpenAI's API later exposes a reliable soft-mode refusal discriminator, does that supersede the strict-schema arc or merely add another admissible shaper?

These questions belong in the feature specs. They should not be answered through incidental implementation choices.

---

## 16. Sources and repository anchors

### Repository

- `DECISIONS.md#056` — OpenAI-compatible providers through host-qualified `HostProfile` records; paid wire + scorecard admission. **The public anchor for everything below** — the two feature specs are gitignored local-only design records (not visible in the public repo); their durable decisions live in `#056` and its amendments.
- `specs/2026-06-27-openai-compatible-providers.md` *(local-only)* — whole-pipeline host selection and deferred provider mixing.
- `specs/2026-07-18-openai-native-host.md` *(local-only)* — GPT-5.6 host design and refusal/proof-boundary gates.
- `src/outrider/llm/host_profiles.py` — host profiles, JSON modes, shaper contract, add-a-host checklist.
- `src/outrider/llm/openai_compatible_provider.py` — request shaping, refusal extraction, usage accounting, and error translation.
- `src/outrider/llm/pricing.py` — host-qualified rate table and GPT-5.6 tier policy.
- `src/outrider/agent/graph.py` — logical graph identities and review-wide host-triad assumptions.
- `src/outrider/agent/nodes/patch_generation.py` — Synthesize patch sub-pass; not a graph node.
- `COST_ANALYSIS.md` — current Outrider cost assumptions and tier/caching analysis.
- `tests/eval/grading.py` — deterministic scorecard grading rather than LLM-as-judge.

### OpenAI

- Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- GPT-5.6 Sol model page: https://developers.openai.com/api/docs/models/gpt-5.6-sol
- API pricing: https://developers.openai.com/api/docs/pricing

### Anthropic

- Claude API pricing: https://platform.claude.com/docs/en/about-claude/pricing

### Research

- https://arxiv.org/abs/2404.13076
- https://aclanthology.org/2024.acl-long.826/
- https://arxiv.org/abs/2504.03846
- https://aclanthology.org/2025.emnlp-main.86/

---

## Final position

Outrider should not choose between "Claude forever" and "force GPT through the current host contract."

The better path is architectural:

- preserve the compatible-provider abstraction;
- route admitted operations independently;
- give GPT a proof-safe response contract;
- use Claude and GPT as independent sources of candidate findings;
- keep deterministic systems and humans in authority;
- spend the second-model budget only where it produces measurable validated value.

That produces a genuine multi-model reviewer rather than a fragile model swap or an expensive debate between two opaque judges.
