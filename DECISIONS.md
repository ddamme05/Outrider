# Decisions

Architectural decision records for Outrider. Each entry documents a decision that code or other docs need to cite — the kind of "why is it this way?" question that would otherwise require digging through commit history or guessing.

## How entries work

Each decision is numbered sequentially starting at 001. **Numbers are never reused**, even if a decision is superseded or abandoned — the number is the stable anchor that code comments reference, and rebinding it would break those references.

Every entry has five sections:

- **Status.** `Accepted, YYYY-MM-DD` for active decisions. When superseded, the line becomes `Superseded by #NNN, YYYY-MM-DD` and the original decision text stays intact below it. Abandoned decisions get `Abandoned, YYYY-MM-DD — <one-line reason>`.
- **Context.** What problem or constraint forced the decision. Usually a few sentences. If there's a spec section that motivated it, cite it here as `spec §X.Y`.
- **Decision.** The actual choice made, stated in the present tense. This is the part code comments cite.
- **Consequences.** What this decision commits the project to — downstream requirements, tradeoffs accepted, patterns that follow from it.
- **Referenced from.** Files in the repo that cite this decision. Updated as new references land; not exhaustive if the decision is widely cited.

## Citation format

Code comments and other docs reference decisions by number and slug:

```
# See DECISIONS.md#001-single-postgres-driver
```

The slug is the lowercased, hyphenated form of the decision's title. If a title changes, the slug changes with it — update the references in the same commit.

## Supersession

When a decision is replaced, the new entry explicitly names what it supersedes:

```
## 007. Celery for durable background execution

**Status:** Accepted, 2026-08-15. Supersedes #003.
```

The old entry's status line becomes `Superseded by #007, 2026-08-15` and the rest of its content is left untouched. A reader investigating why something works the way it does can follow the chain forward from any historical reference.

## Drafting new entries

Per `docs/workflow.md`, Claude drafts entries when a decision needs a stable anchor and the human approves before the entry is committed. Entries do not land without approval. Under the hybrid visibility split, `DECISIONS.md` is part of the public repo — entries are written with an outside reader in mind, concise and self-contained enough that someone with no project history can follow the reasoning.

---

## 001. Single Postgres driver (psycopg3)

**Status:** Accepted, 2026-04-22. Supersedes spec §14.1's two-driver pattern.

**Context.** Spec §14.1 specified two Postgres connection URLs pointing at the same database: `DATABASE_URL` (asyncpg driver, for SQLAlchemy) and `CHECKPOINT_DATABASE_URL` (psycopg3 driver, for LangGraph's `AsyncPostgresSaver`). The split existed because the spec assumed SQLAlchemy 2.0 needed asyncpg for async support while LangGraph's checkpointer required psycopg3, so both had to ship.

Re-examining the constraints: SQLAlchemy 2.0 supports psycopg3 natively through its async engine (`postgresql+psycopg://`). The two-driver pattern buys nothing the single-driver pattern doesn't give, and costs a second binary dependency, a second supply-chain surface, and a second driver's worth of behavior to understand when something goes wrong in production.

**Decision.** Both `DATABASE_URL` and `CHECKPOINT_DATABASE_URL` use the `postgresql+psycopg://` scheme. The two env vars remain separate for documentation clarity and for the option of pointing them at different databases in the future, but they use the same driver in V1.

**Consequences.**
- `pyproject.toml` ships `psycopg[binary]>=3.3.3` and does not list `asyncpg`.
- SQLAlchemy async engine construction uses `create_async_engine("postgresql+psycopg://...")`.
- LangGraph `AsyncPostgresSaver` continues to work unchanged — psycopg3 is its native driver.
- One driver to audit for supply-chain changes, one driver's performance characteristics to learn, one binary wheel in the dependency tree.
- The two env vars still exist so a future decision can split databases without a breaking config change.
- Spec §14.1's documentation of "two drivers, one database" is superseded. Anywhere the spec references asyncpg in deployment, read as psycopg3.

**Referenced from:** `pyproject.toml`, `src/outrider/db/engine.py` (when written), `.env.example`.

---

## 002. Spec lives in Markdown at `docs/spec.md`

**Status:** Accepted, 2026-04-22.

**Context.** The original V1 specification was authored in LaTeX and rendered to a 48-page PDF. LaTeX produces clean typography but is hostile to the workflow the project actually needs: the spec has to be grep-able, diffable in pull requests, citable by line and heading from code comments, and — critically — readable by `scripts/extract_invariants.py`, which parses HTML comment blocks inline in the spec to generate `docs/invariants.md`. LaTeX makes the inline-comment mechanism awkward, and PDF makes it impossible.

Anchor citations are the deciding factor. `docs/spec-index.md` references sections by identifier (e.g., "spec §4.1.6: hitl"), and the extracted docs (`docs/trust-boundaries.md`, `docs/architecture.md`) cite the spec by section number in the same format. In Markdown those anchors are live links into a single file; in PDF they're coordinates on a page that token-costly extraction tooling has to resolve on every lookup.

**Decision.** The spec is converted once to Markdown at `docs/spec.md` and that file becomes the sole source of truth. TikZ diagrams become Mermaid. Code listings become fenced code blocks with language tags. Section numbering from the original PDF is preserved in heading text so existing citations in `docs/spec-index.md` stay valid. The Markdown version is canonical — any divergence is resolved by editing the Markdown. If a PDF is needed for interview sharing, it is generated from the Markdown or kept outside the repo as a derivative artifact.

The conversion commit is purely mechanical — faithful transcription, no "while I'm here" edits. Subsequent corrections go through `docs-only:` commits or `DECISIONS.md` supersessions.

**Consequences.**
- `grep`, `git diff`, and `git blame` all work on the spec the same way they work on code.
- `scripts/extract_invariants.py` can read the spec as a flat text file and the `<!-- invariant:id=... -->` tagging mechanism (see #004) works natively.
- `docs/spec-index.md` citations become live Markdown anchors rather than PDF page references.
- Complex typography (multi-column tables, precise figure placement) is no longer available. The spec doesn't need it.
- Any PDF used for interview sharing is treated as a derivative artifact — if the Markdown and the PDF ever diverge, the Markdown wins.

**Referenced from:** `docs/spec-index.md`, `scripts/extract_invariants.py`, `docs/invariants.md`, `.pre-commit-config.yaml` (the `invariants-in-sync` hook reads `docs/spec.md` directly).

---

## 003. CLAUDE.md is canonical; no AGENTS.md

**Status:** Superseded by #010, 2026-04-22. Originally accepted 2026-04-22, superseding spec §14.2's CLAUDE.md + AGENTS.md sync pattern.

**Context.** Spec §14.2 prescribed shipping two root-level agent-context files, `CLAUDE.md` and `AGENTS.md`, kept in lockstep by a pre-commit divergence check. The rationale was that `CLAUDE.md` is what Claude Code reads while `AGENTS.md` is the vendor-neutral convention for Codex, Cursor, Aider, and other coding agents. Two files exist because clients look them up by name and symlinks break on Windows and in some clients.

The spec's rationale is sound *when multiple agents are in use*. This project is developed solo with Claude Code only. `AGENTS.md` would be dead weight — a second identical file to maintain, a pre-commit hook to enforce the duplication, and a silent-overwrite hazard if either file gets edited directly without the other.

**Decision.** Ship `CLAUDE.md` only. Do not create `AGENTS.md`. Do not ship a divergence-check pre-commit hook. If a future contributor starts using a non-Claude agent, they can add a one-line `AGENTS.md` that reads `See CLAUDE.md` — better than maintaining two identical files.

**Consequences.**
- Root directory is `CLAUDE.md` alone. `AGENTS.md` does not exist.
- `.pre-commit-config.yaml` has no `claude-agents-sync` hook.
- Spec §14.2's two-file pattern is superseded. Anywhere the spec references `AGENTS.md` alongside `CLAUDE.md`, read as `CLAUDE.md` only.
- If the project later adopts a non-Claude agent, this decision gets superseded rather than amended — a new entry would describe the new setup (likely one-line shim or full re-sync) and cite this one.

**Referenced from:** root directory layout, `.pre-commit-config.yaml`, `docs/workflow.md`.

---

## 004. Invariants are tagged inline in the spec, extracted to `docs/invariants.md`

**Status:** Accepted, 2026-04-22.

**Context.** The spec contains roughly 25 architectural rules that must always hold — the LLM never sets severity, audit events are immutable, nodes receive dependencies via closure, no vendor SDK imports outside wrapper folders. These rules need to be visible to future contributors (human and AI) at the moment they write code, not buried in a 48-page spec nobody rereads. Two obvious but wrong options: (a) duplicate the rules into a separate hand-maintained file, which guarantees drift the first time someone edits one place and forgets the other; (b) trust that contributors remember the rules, which has a predictable failure mode.

**Decision.** The rules are tagged inline in `docs/spec.md` using HTML comment blocks, and `docs/invariants.md` is generated from those tags by `scripts/extract_invariants.py`. The spec is the single source of truth; the catalog is a derived view.

Tag format:

```
<!-- invariant:id=severity-set-by-policy
     rule: The LLM never sets finding severity...
     violation: FindingSeverity assigned from LLM output directly...
     check: (optional) grep command that verifies compliance
     security: critical (optional; flags security-critical invariants)
-->
```

Structural rules enforced by the extractor, all as hard failures:
- IDs are unique across the entire spec.
- No tag field contains a hardcoded section reference (see #005).
- Every tag sits inside a numbered H2/H3/H4 heading's scope; tags in the preamble fail extraction.
- Tags inside fenced code blocks are silently skipped, so example tags in documentation don't get treated as live invariants.

The extractor also supports a `--check` mode that exits non-zero if the committed `docs/invariants.md` is out of sync with what re-extraction would produce, enabling the pre-commit guard (see #008).

**Consequences.**
- Editing a rule is always a one-file change in `docs/spec.md` followed by `.venv/bin/python scripts/extract_invariants.py` to regenerate the catalog.
- The generator rejects structural mistakes (duplicate IDs, missing required fields, embedded section references) with named errors pointing at the offending tag's byte offset.
- `docs/invariants.md` is never hand-edited. The file's own header makes this explicit: "Do not edit directly. To change an invariant, edit the invariant tag block in the cited spec section and regenerate this file."
- The invariant catalog becomes usable by downstream tooling — the `outrider-navigator` skill (see #009) reads it to surface applicable rules at the start of each task.
- Three entries are preserved as *forwarding stubs* rather than real invariants (`github-token-scope-minimum-viable`, `llm-output-is-untrusted`, `prompt-caching-always-on`) because they're either deployment rules, umbrella framings, or performance conventions rather than code-bug-detectable rules. The stubs remain in `docs/invariants.md` for citability, with pointers to the actual rule location.

**Referenced from:** `scripts/extract_invariants.py`, `scripts/test_extract_invariants.py`, `docs/invariants.md` (generated), `docs/spec.md` (source), `.pre-commit-config.yaml`, `.claude/skills/outrider-navigator/SKILL.md`.

---

## 005. Invariant source references are derived, never embedded in tags

**Status:** Accepted, 2026-04-22.

**Context.** Early drafts of the invariant tag schema included a `source: §7.4` field where the author declared the spec section the rule came from. The problem surfaced immediately: spec section numbers change. When §7.3 gets a new subsection, everything below shifts by one, and every tag with an embedded `source: §7.x` reference becomes silently wrong. This is exactly the kind of drift the extraction pipeline is built to prevent — and having it *inside* the invariant mechanism itself would undermine the trust claim the whole pipeline exists to support.

**Decision.** Tags never contain a section reference. At extraction time, `scripts/extract_invariants.py` walks backward from each tag's byte offset to find the nearest numbered H2/H3/H4 heading and uses that heading's number and title as the Source. A regex at extraction time rejects any tag field containing `§N.N` or `section N.N.N` with a named error pointing at the offending tag.

Cross-references between invariants use the invariant's ID (stable) rather than its section (not stable). The navigator skill and the stub-forwarding entries both follow this rule.

**Consequences.**
- Spec reorganization — adding sections, renumbering, splitting one section into two — never breaks invariant metadata. The tags move with their sections and the Source updates automatically on the next extraction.
- Tags must physically sit inside the section whose rule they state. Moving a tag into the preamble or into a different section is a hard extraction failure with a clear "not inside any numbered H2/H3/H4 section" message.
- The regex that rejects embedded section references carefully excludes legitimate decimal values (`0.9`, `0.75`, `Python 3.13`). This is tested explicitly in `scripts/test_extract_invariants.py` as a load-bearing case — the `confidence-is-computed-not-assigned` invariant literally contains `0.9, 0.75, 0.5` in its rule field.
- The tradeoff is that section numbers are only ever visible in the *generated* `docs/invariants.md`, not in the source tags. When a reader wants to see the source context, they follow the Source reference in the generated file back to the matching heading in `docs/spec.md`.

**Referenced from:** `scripts/extract_invariants.py` (SECTION_RE and HEADING_RE), `scripts/test_extract_invariants.py` (the decimal-is-not-a-section case).

---

## 006. Two Month 0 spikes, not five

**Status:** Accepted, 2026-04-22. Supersedes spec §15.1's five-spike plan.

**Context.** Spec §15.1 prescribed five throwaway Month 0 spikes: tree-sitter, GitHub App + webhook tunnel, GitHub line mapping, import resolution, and prompt cache. The rationale was that each spike retires a technical unknown before it becomes load-bearing.

Re-examining each: (a) the prompt cache is not a spike — it's a flag on the Anthropic API call, validated by reading `cache_read_input_tokens` the first time `AnthropicProvider.complete()` runs; (b) GitHub line mapping is not a spike — it's coordinate math that belongs in `coordinates/` with thorough unit tests, and the way to prevent off-by-one demo embarrassment is good tests on `coordinates/translator.py`, not a pre-build investigation; (c) import resolution folds into building `ast_facts/` — what's resolvable and what isn't gets discovered while writing `python_adapter.py`, and if it turns out harder than expected, V1 trace scope gets downgraded to same-file-only per the spec's existing fallback.

The two remaining spikes address genuine unknowns that do need resolution before they become load-bearing.

**Decision.** Month 0 ships two spikes, both as throwaway code under `spikes/`:

1. **Tree-sitter spike (1–2 weekend-days).** Parse a Python file, extract function definitions with line ranges, map a diff line range to the containing function, confirm S-expression queries behave as documented. Output: `spikes/tree_sitter/NOTES.md` answering the specific questions that gate the real `ast_facts/` build.
2. **GitHub App + smee.io spike (1 weekend-day).** Register a GitHub App on a test repo, sign a JWT, mint an installation token, receive a webhook payload via smee.io, verify the signature, log the payload shape. One FastAPI route. Output: `spikes/github_app/NOTES.md` with the installation ID, webhook event shapes, and the `githubkit` method names that worked.

The three dropped spikes are handled in-line during the real build instead of upfront.

**Consequences.**
- Month 0 finishes in 2–3 weekend-days instead of 5–7. The build sequence from spec §15.2 onward starts sooner.
- The risk that a dropped spike reveals a hard unknown late (e.g., a tree-sitter behavior that breaks trace assumptions) is accepted. Mitigation: the spec's existing graceful-degradation rules (parse-failed files degrade to LLM-only review, import resolution limits trace to same-file when cross-file fails) cover the known failure modes.
- `spikes/` is a real directory in the repo and its contents survive as documentation after the real build lands. Nothing under `spikes/` is shipped; it's reference material for future readers who want to see what was tested and what was learned.
- Spec §15.1's five-spike framing is superseded. Anywhere the spec references the five-spike plan, read as the two-spike plan above.

**Referenced from:** `spikes/tree_sitter/NOTES.md` (when written), `spikes/github_app/NOTES.md` (when written), `ITERATION_LOG.md` (Month 0 entries).

---

## 007. smee.io for the Month 0 webhook tunnel, not cloudflared

**Status:** Accepted, 2026-04-22. Supersedes spec §6.6's cloudflared recommendation for the spike phase only.

**Context.** Spec §6.6 recommends Cloudflare named tunnels as the primary local-development webhook tunnel, with smee.io listed as a fallback. The rationale is that a named tunnel gives a permanent URL surviving restarts, which matters when the registered GitHub App webhook URL shouldn't change across dev sessions.

For a Month 0 spike lasting 2–3 days, that constraint is weaker than the setup cost. Cloudflare named tunnels require a Cloudflare account, `cloudflared` installation, and named tunnel creation. smee.io is zero-setup: one `npx smee-client` command, channel URL generated instantly, officially recommended by GitHub for local webhook testing. The smee channel URL persists across client restarts within a reasonable window, which is all the spike needs.

The spec's concerns apply later — when the GitHub App is installed on the real demo repo and the webhook URL needs to stay stable across weeks of development. At that point, the tunnel choice is revisited.

**Decision.** The GitHub App spike (#006) uses smee.io for webhook tunneling. The production deployment (Month 3 onward) uses whatever the team has set up — typically the cloud hosting's public URL, but Cloudflare named tunnels remain a valid local-dev option if a developer wants the persistent-URL property.

The scope of this decision is Month 0 only. It does not override spec §6.6's general guidance; it narrows it for the spike phase.

**Consequences.**
- `spikes/github_app/NOTES.md` documents the smee.io setup steps and the commands actually run.
- If the spike client disconnects, webhooks sent while it's down are lost (smee doesn't buffer). This is accepted for spike use.
- Post-Month-0 deployment choice is deferred and will get its own decision entry if it supersedes spec §6.6 in any meaningful way.

**Referenced from:** `spikes/github_app/NOTES.md` (when written), `docs/workflow.md` (spike setup section).

---

## 008. Pre-commit is the primary drift gate, CI is secondary

**Status:** Accepted, 2026-04-22.

**Context.** `scripts/extract_invariants.py --check` detects drift between `docs/spec.md` and `docs/invariants.md`, but the check only has value if it runs before problematic commits land. CI-only enforcement catches drift at PR time — after the author has moved on, after the problematic commit is already in git history, and every failure requires an additional push that pollutes PR history. Local enforcement at commit time catches drift in the same context where it was introduced.

The same reasoning applies to `scripts/test_extract_invariants.py`: the extractor's output is only trustworthy when its tests pass, and a broken extractor edit that silently emits wrong invariants is a worse failure mode than a PR that fails CI.

**Decision.** Ship `.pre-commit-config.yaml` at the repo root with local hooks that use the pinned project virtualenv. Ruff and Ruff format run first so formatting doesn't invalidate extraction:

1. **`ruff`** runs `.venv/bin/ruff check --fix`.
2. **`ruff-format`** runs `.venv/bin/ruff format`.
3. **`invariants-in-sync`** runs `.venv/bin/python scripts/extract_invariants.py --check` and is scoped via `files:` to fire only when `docs/spec.md`, `docs/invariants.md`, or `scripts/extract_invariants.py` changes. Zero overhead on unrelated commits.
4. **`extractor-tests-pass`** runs `.venv/bin/python scripts/test_extract_invariants.py` and is scoped to fire only when the extractor or its test file changes.
5. **`decision-refs-resolve`** runs `.venv/bin/python scripts/check_decision_refs.py` and is scoped to DECISIONS.md or the checker script.

The hooks use `language: system` because the repo already pins the toolchain in `uv.lock` and installs it into `.venv` via `uv sync --dev`. This avoids network fetches during hook execution and avoids relying on a `python` shim that may not exist on every machine.

CI runs the same checks as a secondary gate. If a contributor skips `pre-commit install`, CI catches the drift at PR time — this is acceptable as a fallback but not the primary defense.

**Consequences.**
- Contributors running `pre-commit install` after cloning get both guards automatically. The README documents this setup step.
- A commit touching `docs/spec.md` without regenerating `docs/invariants.md` is rejected locally with a clear message pointing at the fix command.
- A commit modifying `scripts/extract_invariants.py` that breaks the test suite is rejected locally. `scripts/test_extract_invariants.py` exits non-zero on failure for exactly this reason — earlier versions printed `[PASS]`/`[FAIL]` lines but always exited 0, which would have made the hook a rubber stamp.
- Hook ordering matters. `ruff` then `ruff-format` both run before the extraction hooks so the spec is fully lint-fixed and reformatted before extraction reads it. Any future generator hook (e.g., a table-of-contents autogenerator) goes between `ruff-format` and `invariants-in-sync` so the sync check sees the final post-generation spec.
- A commit that touches `docs/spec.md`, `docs/invariants.md`, and `scripts/extract_invariants.py` simultaneously triggers both hooks at once. Good friction — that's the commit most likely to introduce drift, and requiring all three constraints to hold simultaneously forces internal consistency.

**Referenced from:** `.pre-commit-config.yaml`, `scripts/extract_invariants.py` (the `--check` mode this hook relies on), `scripts/test_extract_invariants.py` (exit-code behavior this hook relies on), `README.md` (setup instructions).

---

## 009. Invariants surface via the navigator skill, not via CLAUDE.md inlining

**Status:** Accepted, 2026-04-22.

**Context.** Once `docs/invariants.md` existed, the question was how to get the right invariants in front of Claude at the moment code gets written. The naive option is to inline the entire catalog into `CLAUDE.md` — all 25 rules visible on every session regardless of what's being worked on. This is the "dump everything" pattern: guaranteed coverage, useless in practice because the user stops reading the top matter after the third session and the signal drowns in itself.

A second tempting option — keyword-matching from the user's message to invariant IDs — is fragile. Keyword mismatch means missed invariant, and the skill itself shouldn't need to be clever about synonym expansion or phrasing tolerance.

**Decision.** Ship `outrider-navigator` as a skill at `.claude/skills/outrider-navigator/SKILL.md`. The skill follows a deterministic section-mapping workflow:

1. Identify which spec section the current task touches, using explicit mapping rules for the 7 graph nodes, the `ast_facts` and `coordinates` modules, the audit subsystem, the LLM provider, the HITL path, and the webhook receiver.
2. Load `docs/invariants.md` and select entries whose `Source.` field matches that section, plus categorical "always apply" buckets: vendor SDK / state purity rules always apply to any code change; `[security-critical]` entries always apply to webhook/path/shell surfaces; schema-enforcement rules always apply to audit-event and finding changes.
3. Surface them at the start of the response in a fixed format (`Invariants in scope (from docs/invariants.md): ...`) before any code, giving the user a scannable checkpoint to catch missed invariants before 200 lines of wrong code land.
4. On conflict between user request and an invariant, name the conflict explicitly, cite the invariant, and offer compliant alternatives rather than quietly accommodating the violation.

The skill triggers on code-producing work only. Conceptual and planning questions don't trigger it — the friction of an unsolicited invariant dump during discussion outweighs the benefit. This boundary is held provisionally for V1 and will be revisited after the first real test-drive.

**Consequences.**
- Invariants become visible at the precise moment they can prevent a bug, not before and not after. The fixed output format gives the user a standard checkpoint to scan before each response.
- The skill's triggering accuracy depends on the `description` field being pushy enough; undertriggering is a real failure mode (spec note from skill-creator conventions). The first real task run through the skill is the validation.
- The skill does not *enforce* invariants — enforcement lives in the extractor's `--check` mode (#008), in Pydantic schema validators, and in the grep-based `Check` commands embedded in each invariant entry. The navigator's job is visibility, not gating.
- The skill does not rewrite `docs/invariants.md`. Changes go to the `<!-- invariant:id=... -->` block in `docs/spec.md` and flow through the extractor per #004.
- The umbrella-stub entries (`llm-output-is-untrusted`, `github-token-scope-minimum-viable`, `prompt-caching-always-on`) are handled by an explicit rule: the skill cites the pointer target or the named child invariants, never the stub itself.

**Referenced from:** `.claude/skills/outrider-navigator/SKILL.md`, `docs/invariants.md` (the catalog the skill reads). `CLAUDE.md` does not currently reference the skill — the skill autoloads from `.claude/skills/` and its metadata goes into context without an explicit mention. If a future `CLAUDE.md` revision adds a pointer (e.g., in the "Architecture and conventions" section), add it here.

---

## 010. Codex is the auditor/reviewer; AGENTS.md and repo-level Codex skills exist for that purpose

**Status:** Accepted, 2026-04-22. Supersedes #003.

**Context.** Decision #003 said "Ship `CLAUDE.md` only. Do not create `AGENTS.md`." The reasoning was that maintaining two identical agent-context files for a solo-Claude-Code workflow was dead weight. That reasoning held while Claude Code was the only agent in the workflow.

The workflow has changed. OpenAI Codex is now part of it, with a specific and narrow role: auditor/reviewer. Codex does not write feature code, does not draft specs, and does not update tracking logs — Claude Code owns all of that. Codex reads diffs, evaluates them against the project's trust boundaries and invariants, and flags problems. Two agents, two non-overlapping roles.

Under this arrangement, `AGENTS.md` and `CLAUDE.md` are no longer duplicates. They are *different entry points with different contents serving different roles*:

- `CLAUDE.md` is Claude Code's entry point. It carries the implementer context: architecture, conventions, workflow rules, spec navigation, testing. It points Claude Code at the docs an author needs.
- `AGENTS.md` is Codex's entry point. It carries the reviewer context: a pointer to the invariant catalog as a checklist, a pointer to the trust-boundary map as a scrutiny guide, MCP usage rules for documentation lookups during review, and the project-specific reviewer skill.

This is the distinction #003 didn't contemplate. When the question was "ship two identical files to appease tool discovery," the answer was correctly no. When the question is "ship two files with genuinely different role-specific contents," the answer is yes — the files aren't duplicates, they're complements.

A second factor re-examined from #003: Codex's discovery paths. Codex reads `AGENTS.md` at project root (merged with any `~/.codex/AGENTS.md` global file) and loads skills from `~/.codex/skills/` at the user level or `.agents/skills/` at the repo level. These are separate trees from `.claude/skills/`. Repo-level Codex skills therefore land at `.agents/skills/`, not under `.claude/`. The exact path should be re-verified against the installed Codex CLI version before authoring begins, since OpenAI's skill discovery conventions have shifted at least once; if the current CLI uses a different repo-level path, apply it and note the correction in a follow-up decision if the shape of the fix is architecturally meaningful.

**Decision.** Codex occupies the reviewer role. The repository ships both `AGENTS.md` and `CLAUDE.md` as root-level agent-context files with distinct, role-specific contents:

- `CLAUDE.md` remains the canonical Claude Code entry point. Contents unchanged from the current state — implementer-role context.
- `AGENTS.md` is the canonical Codex entry point. Contents are reviewer-role: short preamble, pointer into `docs/invariants.md` as review checklist, pointer into `docs/trust-boundaries.md` as scrutiny guide, the `docs/mcp-usage.md` rules for doc lookups, pointer into the Outrider reviewer skill.

An Outrider-specific reviewer skill is checked into the repo at the Codex repo-level skill path (verified per the context note above). The skill surfaces the project's invariants and trust boundaries as a review checklist and encodes any Codex-specific operational rules that don't belong in `AGENTS.md` itself.

Trail of Bits' `differential-review` and `audit-context-building` skills are installed at the user level via their published install method, targeting `~/.codex/skills/`. They are not checked into the Outrider repo. They provide generic review technique; the Outrider reviewer skill provides project-specific checklist context. The two layers compose: ToB skills handle "how to review a diff," the Outrider reviewer skill handles "what Outrider-specific things to check."

`agentic-actions-auditor` is explicitly *not* part of this install. It audits GitHub Actions workflows for prompt-injection attack vectors — it's a static analyzer for `.github/workflows/*.yml` files, not Python code. If Outrider eventually ships a GitHub Actions integration (V2, per spec §16), `agentic-actions-auditor` becomes relevant for auditing *that workflow file*, not Outrider itself. Revisit at that point.

**Consequences.**

- `AGENTS.md` is tracked in the repo with reviewer-role contents distinct from `CLAUDE.md`. The current `AGENTS.md` on disk (which mirrors `docs/mcp-usage.md` verbatim) is replaced.
- An Outrider reviewer skill is authored at the Codex repo-level skill path and checked in. The skill loads `docs/invariants.md` and `docs/trust-boundaries.md` at review time and produces review-oriented output, not scaffolding-oriented output. It is the reviewer counterpart to `outrider-navigator` (#009) but with distinct output shape.
- `.gitignore` currently ignores `AGENTS.md` (deferred-public state). Under this decision, `AGENTS.md` is tracked from the same moment `CLAUDE.md` and `docs/` become tracked — i.e., at the public-flip point, not now. The ignore list stays as-is until the flip; the flip checklist gains "un-ignore `AGENTS.md` and the Codex repo-level skill path" alongside the existing items.
- The `outrider-navigator` skill (#009) is unaffected. Navigator serves Claude Code at authoring time; the reviewer skill serves Codex at review time. They read the same invariant catalog but produce different output.
- ToB skill installation is a user-level setup step. The exact install command lives in the reviewer-setup documentation (home to be chosen between `docs/deployment.md` or a new `docs/reviewer-setup.md`), not in this decision text, so that if ToB changes their install method later the fix is a doc edit rather than a decision supersession.
- Codex's MCP usage rules currently live in `docs/mcp-usage.md` (also consumed by Claude Code). Under this decision, `docs/mcp-usage.md` remains shared — both agents need the same MCP hygiene. `AGENTS.md` points to it; `CLAUDE.md` already points to it. No duplication.
- #003's concern (maintaining two identical files) does not recur because the files are not identical and their content is not mechanically synced. Each evolves independently as its role's requirements evolve. The pre-commit `claude-agents-sync` hook proposed in the spec's original §14.2 is not reintroduced — it would enforce identity between files that are no longer supposed to be identical.
- If Codex is removed from the workflow in the future, this decision gets superseded rather than amended. A new entry would document the revised setup (likely: delete `AGENTS.md`, delete the reviewer skill, reactivate #003's framing).

**Referenced from:** `AGENTS.md` (once rewritten), the Outrider reviewer skill (once authored), `.gitignore` (public-flip checklist), `docs/workflow.md` (reviewer-role description, to be added when the public-flip doc update happens), `README.md` or `docs/reviewer-setup.md` (ToB install command, same timing).