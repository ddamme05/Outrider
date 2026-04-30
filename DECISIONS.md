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

---

## 011. Self-hosted is canonical V1; SaaS is V1.5+

**Status:** Accepted, 2026-04-23.

**Context.** User-data ownership is a first-class project requirement. Two deployment shapes are possible for an agentic PR reviewer: self-hosted (the user runs Outrider in their own infrastructure) and SaaS (we host, user code flows through our servers). The ownership claim is technically enforceable under self-hosted: code never leaves the user's infra except for the LLM API call (see #013). Under SaaS the same claim becomes a trust commitment rather than a technical property, and would require tenant isolation, public data-handling commitments, and potentially BYOK/enclave architecture before it could be made credibly. None of that is feasible inside V1's build budget (spec §15) without displacing load-bearing feature work.

**Decision.** V1 ships as a self-hostable application. Canonical deployment is: the user runs Outrider in their own infrastructure, connects it to their GitHub App installed on their repos, and points it at Anthropic (or, V1.5+, an alternate provider behind the `LLMProvider` Protocol). The only third-party egress is the LLM API call. V1 does not support multi-tenant SaaS operation. SaaS is deferred to V1.5-or-later and, if adopted, requires a supersession decision covering tenant isolation, data-handling commitments, and operator access controls.

**Consequences.**

- **Explicit egress exception.** Code content does reach one third party by design: Anthropic (or, V1.5+, an alternate `LLMProvider`). The LLM call is where review reasoning happens; there is no self-hosted alternative in V1. DECISIONS.md #013 defines the exact data-handling contract for that egress. "Users own their data" under self-hosted V1 means "your code stays in your infra, modulo the specific LLM egress documented in #013." Anywhere the claim is reproduced in user-facing documentation, the Anthropic qualifier must be present.
- Single-tenant architecture. `reviews`, `audit_events`, and `installations` tables do not carry a `tenant_id`; the deployment *is* the tenant.
- Authentication: `ADMIN_API_KEY` plus the GitHub App installation. No per-user auth in the dashboard.
- Documentation ships self-hosting as the primary path — Docker compose or similar, env-var configuration, secrets-manager integration notes. The README leads with "run this in your infra," not with a sign-up flow.
- SaaS-only features (per-tenant dashboards, org-level billing, cross-installation analytics) are out of scope for V1.
- Retention and uninstall-purge are under user control per DECISIONS.md #012.
- V1 privacy statement becomes: "Outrider runs in your infrastructure. Outrider maintainers do not have access to your code or review data. The only third-party egress is the LLM provider you configure (see DECISIONS.md #013 for the exact data-handling contract)." This is a verifiable claim because we do not run any hosted component.
- The SaaS migration decision, when it happens, must explicitly address every gap category flagged in the 2026-04-23 Codex security audit (tenant isolation, data-handling commitments, operator access controls, BYOK feasibility).

**Referenced from.** `README.md`, `docs/deployment.md`, `docs/architecture.md`.

---

## 012. Data retention: TTLs configurable, purge on installation.deleted

**Status:** Accepted, 2026-04-23.

**Context.** Spec §8 defines `audit_events` as append-only by database trigger — correct for integrity and replay equivalence. Unbounded retention creates a privacy problem under #011: the operator is the user, and users have legitimate retention rights over their review data. GitHub's `installation.deleted` webhook is the canonical "this user has uninstalled Outrider" signal and must trigger a real retention decision rather than silent indefinite retention.

**Decision.** Retention policy for V1:

- Every review, finding, audit event, and installation row has a retention TTL, set in configuration and operator-overridable via `pydantic-settings`.
- On the `installation.deleted` webhook, all reviews / findings / audit events scoped to the installation are marked for deletion. A grace period begins (reinstalling the App within the window restores the tombstone); after expiry, the rows are hard-deleted.
- `installation.suspend` and `installation.unsuspend` do NOT trigger purge. Suspension is reversible on GitHub's side; treat as pause-only.
- Raw webhook payloads are not stored at rest. Only shape-parsed fields persist on audit events.
- The purge job runs as a privileged Postgres role that can bypass the append-only trigger for this one narrow case. Its operations are logged to a separate `purge_audit` table, which records installation_id, rows_affected, timestamp, and the role that performed the purge — no payload content, so the table stays size-bounded.
- Operators who need longer retention (compliance, forensics) can raise the TTLs or disable the purge entirely via configuration. Default is purge-on because the default must favor less-data-held outcomes.

Specific default TTL values (reviews, audit events, installation grace window) are initial defaults, not firm commitments — they live in code configuration and are captured in local ITERATION_LOG rather than embedded in this decision. The retention-policy *shape* (TTLs exist; purge on uninstall with grace window; raw payloads not stored) is the stable anchor; specific numbers may shift in minor updates without requiring supersession.

**Consequences.**

- Spec §8's "append-only by database policy" claim becomes "append-only within retention window, with a privileged purge role for expiry." Requires a docs-only update to `docs/trust-boundaries.md` section 7 at public flip.
- New module `sweep/purge_expired.py` runs on the same APScheduler cadence as the existing sweep jobs (spec §9.4). Uses advisory locks per the `sweep-jobs-use-advisory-locks` invariant.
- New table `purge_audit` records every deletion with installation_id, rows_affected, timestamp, and the role that performed it. Append-only forever; not itself subject to retention TTL, because purge records are the forensic trail operators rely on when investigating accidental data loss.
- `api/webhooks/router.py::installation_deleted_handler` enqueues the purge. Immediate delete risks accidental-uninstall data loss without the grace window.
- Replay equivalence (spec §8) holds within the retention window: any review not yet purged can be replayed under its original policy version.
- Users get a clear retention commitment when deciding whether to install Outrider on sensitive repos. The specific default numbers are visible in configuration and can be raised for compliance use cases.

**Referenced from.** `docs/deployment.md`, `docs/trust-boundaries.md`, `sweep/purge_expired.py` (when written), `db/triggers/audit_append_only.sql` (when written), `api/webhooks/router.py` (when written), `db/models/purge_audit.py` (when written).

---

## 013. LLM/privacy contract: Anthropic egress, retention, ZDR

**Status:** Accepted, 2026-04-23.

**Context.** Under #011 (self-hosted canonical V1), user code stays in user infra *except* for the LLM call to Anthropic. Users deciding whether to install Outrider need a clear statement of what leaves their infrastructure, what Anthropic does with it, and what configuration controls exist. Anthropic's standard API contract (per privacy.anthropic.com as of 2026-04-23): inputs/outputs are not used for training without permission; 30-day retention for abuse detection; zero-data-retention available by separate arrangement. Prompt caching is ZDR-compatible with short-lived cache entries.

**Decision.** The LLM privacy contract Outrider commits to, in V1:

1. **Egress include list.** API calls to the LLM provider include: file contents of changed files, PR title, PR body, commit messages, branch names, author login, scope unit text (extracted via `ast_facts`), and evidence-span snippets for findings. That is the minimum set needed for structure-aware review per spec §5.
2. **Egress exclude list.** API calls do NOT include: GitHub App private key, webhook secret, installation tokens, webhook signatures, operator env vars, or any secret material. If a future code path would send any excluded item, that's a bug *and* a #013 supersession.
3. **Anthropic retention.** 30 days for abuse detection by default; inputs not used for training; zero-data-retention available by customer arrangement. Outrider does not enable ZDR by default — ZDR requires a separate agreement with Anthropic and is not universally available. Operators who have arranged ZDR opt in via `ANTHROPIC_ZDR_ENABLED=true`, which causes `AnthropicProvider` to send the appropriate request header and restrict model selection to ZDR-supported variants.
4. **Prompt caching.** Enabled by default per the existing `prompt-caching-always-on` convention. Prompt cache entries are short-lived (TTL set by Anthropic, currently ~5 minutes as of 2026-04-23) and ZDR-compatible. The TTL is Anthropic-controlled and may change; this decision commits to *using* caching with whatever TTL Anthropic provides, not to a specific duration.
5. **Logs never contain prompt or completion content.** Structured log fields on `LLMCallEvent` are: `review_id`, `model`, `provider`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `elapsed_ms`, `finish_reason`. Never the prompt text. Never the completion text. If debugging requires the content, retrieve it by replay from `PRContext` + scope extraction — the audit row intentionally does not carry it. Defense in depth: a logging filter rejects records containing prompt/completion payloads (first line); the `LLMCallEvent` schema itself omits content fields (belt).
6. **User-facing privacy statement.** "Outrider sends the following to your configured LLM provider: [egress include list above]. Under Anthropic's default API terms, inputs are retained for 30 days for abuse detection and not used for training. If you need zero-data-retention, arrange it with Anthropic and set `ANTHROPIC_ZDR_ENABLED=true`. Outrider itself adds no retention beyond the retention TTLs configured under DECISIONS.md #012."
7. **Mandatory ZDR-disabled startup notice.** On startup, if `ANTHROPIC_ZDR_ENABLED` is not true, Outrider emits a structured log line at INFO level naming the retention commitment: "privacy_notice anthropic_retention=30d zdr=disabled; set ANTHROPIC_ZDR_ENABLED=true if you have arranged zero-data-retention with Anthropic." The line is emitted at INFO not DEBUG so deployment reviewers cannot miss it, and carries a `privacy_notice` structured field so log-aggregation rules can route it to a visible surface. This closes the silent-default failure mode where an operator deploys Outrider without reading configuration docs, user code flows to Anthropic's 30-day retention, and affected users never find out. The README and deployment docs explicitly reference this notice behavior.

**Consequences.**

- `llm/anthropic_provider.py` reads `ANTHROPIC_ZDR_ENABLED`, sends the appropriate header when set, and fails fast at startup if ZDR is enabled against a model variant that does not support it.
- `llm/anthropic_provider.py` emits the mandatory startup notice on construction when ZDR is disabled.
- `audit/events.py::LLMCallEvent` schema: metadata fields only. No `prompt` or `completion` field.
- Logging configuration includes a filter that drops log records whose values contain prompt/completion payloads. First line of defense; the schema itself is the belt.
- V1.5's `OpenAIProvider` (and any other `LLMProvider` implementation) must publish equivalent retention commitments at wrapper-construction time or Outrider refuses to construct the provider. Each provider is responsible for its own #013-class statement; this entry covers Anthropic.
- Prompt caching stays on by default per existing convention. Documentation explicitly notes the Anthropic-controlled TTL and ZDR compatibility.
- The user-facing privacy statement (item 6) ships in `README.md` and `docs/deployment.md` at the public flip. Any change to the egress field list (item 1) or the retention terms (item 3) requires a supersession decision, not an in-place edit of this entry.

**Referenced from.** `llm/anthropic_provider.py` (when written), `llm/base.py` (when written), `audit/events.py` (when written), `README.md`, `docs/deployment.md`.

---

## 014. Audit events are metadata-only; content purge targets `reviews` and `findings`

**Status:** Accepted, 2026-04-23. Narrows #012 by constraining the purge target.

**Amended 2026-04-29:** point 4 + context paragraph + point 5 + point 6 corrected to align with the audit-events module spec drafting reality. (a) `FindingEvent` field set in point 4 (metadata-only replay reconstruction) and in the context paragraph adds `trace_path` (required for INFERRED findings) — without it, metadata-only replay loses the proof artifact for the structural-by-reference tier, and the proof claim degrades to "OBSERVED only after retention." (b) Point 5's `HITLDecisionEvent.per_finding_decisions[*].reason` reference renames to `decisions[*].reason` to align with the cross-boundary `HITLDecision` type's field name (spec.md §7.4 line 290) — same data, one name through the system. (c) Point 5's "specific length bounds and content guards are implementation-level (schema validators)" phrasing splits into two distinct properties, surfaced by the audit-events spec's test-scenario design: length bounds are mechanically enforced via Pydantic `Field(max_length=...)`; the structural-description-not-code-snippet content rule is author/reviewer discipline (no automated heuristic for "looks like code" exists or is in scope to build). (d) Point 6's "No spec or invariant edit is required" / "spec §8 audit story is untouched" framing narrows: spec §8.2 was intentionally corrected in this same amendment via the FindingEvent and HITLDecisionEvent row updates above; `audit-events-append-only` and `docs/trust-boundaries.md` section 7 still stand as written, and the broader spec §8 audit story (purge target, retention modes, append-only enforcement) remains untouched — only §8.2's field-set wording changed. The audit-events module hasn't shipped, so no migration path is required for any correction. No change to architectural shape or to any other point in this decision.

**Context.** #012 specifies a purge job that runs on `installation.deleted` and hard-deletes `audit_events` rows, violating the live `audit-events-append-only` invariant (docs/invariants.md, Source §8.6: "Audit events are corrected by appending superseding events, never by update or delete"). Reading spec §8.2's V1 event types alongside §7.3's `ReviewFinding` resolves the conflict: the spec already split finding metadata from finding content. `FindingEvent` carries `finding_id`, `finding_type`, `severity`, `file_path`, `line_range`, `dimension`, `finding_content_hash`, `evidence_tier`, `query_match_id`, `trace_path`, `policy_version` — all identifiers, hashes, paths, enum values, and (for INFERRED) the structural-by-reference proof artifact. The actual finding content (`title`, `description`, `evidence`, `suggested_fix`) lives on the `ReviewFinding` Pydantic model and, by the same architectural logic, in a separate `findings` table. #012's original framing was wrong about the purge target; it should have been `reviews` and `findings`, not `audit_events`.

**Decision.** #012's retention/purge mechanism targets `reviews` and `findings` (content tables), not `audit_events` (metadata rows):

1. **`audit_events` stays append-only forever.** No TTL, no purge path, no privileged-role bypass of the trigger. The `audit-events-append-only` invariant is preserved literally. Every event in spec §8.2 is metadata-only by design — audit rows carry identifiers, hashes, counts, paths, and enum values, not user code or prompt/completion content.
2. **Purge targets are content tables.** `reviews`, `findings`, and any V1.5+ content tables carry configurable TTLs and are hard-deleted under #012's retention + grace-window mechanism on `installation.deleted`. `purge_audit` (from #012) logs each purge against its content-table target.
3. **`FindingEvent.finding_id` may reference a purged `findings` row.** Designed property, not a bug: the audit trail says "at time T, a finding with this ID was emitted in this phase." The content is gone per retention; the fact of emission is permanent. Dashboard renders a dangling `finding_id` as "content redacted per retention policy (purged YYYY-MM-DD via installation.deleted)." Stronger audit story than "we kept the whole finding forever" — retention actually deletes user content while preserving the claim that every finding was audited.
4. **Replay equivalence has two modes: full replay and metadata-only replay.** Full replay (content still within retention): reconstructs `ReviewState` including all `ReviewFinding` objects with their content. Metadata-only replay (content purged per retention): reconstructs the event sequence from `audit_events` alone, with `ReviewFinding` reconstructed as stubs carrying only the fields `FindingEvent` records (finding_id, finding_type, severity, file_path, line_range, dimension, finding_content_hash, evidence_tier, query_match_id, trace_path, policy_version). The replay tool indicates which mode applied and refuses to silently produce a hybrid. Eval scenarios in `tests/eval/scenarios/replay/` cover both modes explicitly — this is what spec §8.7 "replay equivalence" now promises under retention.
5. **Borderline fields on audit events are metadata, not content, by convention.** `LLMCallEvent.context_summary` carries `(file_path, scope_unit_name, line_start, line_end)` per spec §8.3 — manifest, not content. `FileExaminationEvent.file_path` — structural identifier. `TraceDecisionEvent.reason` and `HITLDecisionEvent.decisions[*].reason` are free-text fields that must carry structural descriptions, not code snippets. The two properties are enforced differently: length bounds are mechanically enforced via Pydantic `Field(max_length=...)` (e.g., 500 chars); the structural-description-not-code-snippet content rule is author/reviewer discipline (no automated heuristic for "looks like code" is in scope).
6. **Spec §8.2 corrections were made in this amendment; live invariants and trust-boundaries doc are untouched.** spec §8.2's `FindingEvent` and `HITLDecisionEvent` field lists were corrected in lockstep with the Amended 2026-04-29 marker above (FindingEvent gains `trace_path`; HITLDecisionEvent's reason path renames `per_finding_decisions[*]` → `decisions[*]`). `audit-events-append-only` stands as written; `docs/trust-boundaries.md` section 7 stands as written; #012's retention story still narrows to content tables; the broader spec §8 audit story (purge target, retention modes, append-only enforcement) is untouched — only §8.2's field-set wording changed.

**Consequences.**

- #012's "privileged purge role that can bypass the append-only trigger" mechanism is removed. The trigger is absolute: no role DELETEs or UPDATEs on `audit_events`.
- Alembic migration #1 defines `audit_events` with the append-only trigger unchanged; `reviews` and `findings` are separate tables with a `retention_expires_at` column populated at insert time.
- `sweep/purge_expired.py` queries `reviews` and `findings` for rows past `retention_expires_at`; `audit_events` is never in the query.
- The `installation.deleted` handler marks content rows for deletion; grace window, hard-delete at expiry, and `purge_audit` logging behave per #012 — but the target rows are `reviews` and `findings`, not `audit_events`.
- Dashboard gains a rendering rule: a `FindingEvent` whose `finding_id` no longer resolves in `findings` renders as "content redacted per retention" with the purge date from `purge_audit`.
- The replay tool distinguishes full-replay from metadata-only-replay modes and refuses hybrid output. `tests/eval/scenarios/replay/full_replay/` and `.../metadata_only_replay/` scenarios cover both modes.
- #012's specific default TTL numbers (in local ITERATION_LOG) apply to the content tables. `audit_events` has no TTL.
- `FindingEvent.finding_id` is stored on audit rows as a plain UUID, not a foreign key — an FK with ON DELETE CASCADE would violate the append-only guarantee, and ON DELETE SET NULL would mutate audit rows. Plain UUID lets the content purge proceed without touching audit_events.

**Referenced from.** `docs/trust-boundaries.md` section 7 (no edit required; narrow by implication), `sweep/purge_expired.py` (when written), `db/models/reviews.py` (when written; carries `retention_expires_at`), `db/models/findings.py` (when written; same), `db/models/audit_events.py` (when written; no retention column), `api/webhooks/router.py` (when written), dashboard finding-detail renderer (when written), `tests/eval/scenarios/replay/` (when written).

---

## 015. ZDR is an operator attestation, not a runtime opt-in

**Status:** Accepted, 2026-04-23. Narrows #013's ZDR mechanism (points 3 and 7).

**Context.** #013 drafted ZDR as a per-request mechanism — `ANTHROPIC_ZDR_ENABLED=true` would cause `AnthropicProvider` to "send the appropriate request header" and "restrict model selection to ZDR-supported variants." Anthropic's official documentation (platform.claude.com/docs/en/build-with-claude/api-and-data-retention; privacy.claude.com/en/articles/7996866) resolves this differently: ZDR is enabled at the organization level via a contract/sales arrangement with Anthropic, not per-request. There is no API header that toggles ZDR on a per-call basis; model selection is not restricted (the Messages API features Outrider uses are ZDR-eligible whenever the org-level arrangement is in place). The #013 framing would have caused Outrider to emit a non-existent header and given operators a false sense that setting the flag enables ZDR.

Retention terms also corrected: Anthropic's standard retention is **30 days** for all API inputs/outputs by default (not "30 days for abuse detection" as #013 implied). For content flagged for policy violations, retention extends to **2 years** for content and **7 years** for classification scores — and these exceptions apply even under ZDR.

**Decision.** `ANTHROPIC_ZDR_ENABLED` is reframed as an **operator attestation**, not a runtime enablement:

1. **ZDR is enabled by the operator's contract with Anthropic.** Outrider cannot enable ZDR. When the operator has arranged ZDR at the organization level, setting `ANTHROPIC_ZDR_ENABLED=true` attests "I have confirmed my Anthropic organization has ZDR arranged."
2. **No per-request header, no model-variant restriction.** `AnthropicProvider` does not send a ZDR-specific header (none exists). Model selection is unchanged regardless of the flag. The flag's runtime effect is entirely on messaging (startup notice + user-facing privacy statement), not on the API request shape.
3. **Retention terms (corrected from #013 point 3).** Standard: 30 days for all inputs/outputs. Policy-violation: up to 2 years content, 7 years classification. No training use without permission. ZDR narrows standard retention (no data at rest after API response), but policy-violation retention still applies.
4. **Startup notice (revised from #013 point 7).** On startup, `AnthropicProvider` emits a structured INFO log line naming the operator-asserted retention posture:
   - If `ANTHROPIC_ZDR_ENABLED` is not true: `privacy_notice anthropic_retention=30d zdr=not_attested; ZDR cannot be enabled by this flag alone — it requires a contract arrangement with Anthropic (contact sales). Set ANTHROPIC_ZDR_ENABLED=true if your organization has ZDR arranged.`
   - If `ANTHROPIC_ZDR_ENABLED=true`: `privacy_notice anthropic_retention=zdr_attested; ZDR arrangement assumed per operator attestation (Outrider does not verify). Policy-violation retention up to 2 years content / 7 years classification still applies per Anthropic's ZDR terms.`
   Both lines emit at INFO, not DEBUG, so operators and deployment reviewers see them on every worker start.
5. **User-facing privacy statement (revised from #013 point 6).** The statement in README and deployment docs becomes: "Outrider sends to the configured LLM provider: file contents of changed files, PR title, PR body, commit messages, branch names, author login, scope unit text, and evidence-span snippets (see #013 for the full egress include list). Under Anthropic's default terms, inputs are retained for 30 days; content flagged for policy violations is retained up to 2 years, classification scores up to 7 years; no data is used for training without permission. If your organization has a ZDR arrangement with Anthropic, set `ANTHROPIC_ZDR_ENABLED=true` — Outrider uses the flag to adjust its startup notice and privacy statement; it does not enable ZDR on its own. Contact Anthropic sales if you need to arrange ZDR. Outrider does not currently support HIPAA-subject workloads."
6. **Verification is out of scope for V1.** Outrider does not call an admin endpoint to verify ZDR status, because Anthropic does not document one. The operator is trusted to report their own arrangement. A V1.5+ verification path may become possible if Anthropic exposes ZDR status on a dedicated endpoint; revisit then.
7. **HIPAA readiness is not supported in V1.** Anthropic offers HIPAA readiness as an alternative to ZDR (organization-level BAA, enforces feature restrictions that return 400 on ineligible features). Outrider's own data handling (`reviews` and `findings` tables under #014, `audit_events` metadata) has not been assessed for HIPAA composability in V1. Operators running HIPAA-subject workloads should not install Outrider on repositories containing PHI until a V1.5+ decision covers (a) Outrider's own PHI handling, (b) Anthropic's HIPAA configuration as seen through `AnthropicProvider`, (c) audit-log handling of any incidental PHI exposure via code content reaching the LLM. Not a silent deferral: the user-facing privacy statement (point 5) names this explicitly — "Outrider does not currently support HIPAA-subject workloads" — so an enterprise evaluator asking "can this run on BAA-subject workloads?" gets an unambiguous "no, not in V1" rather than having to infer it.

**Consequences.**

- `llm/anthropic_provider.py` reads `ANTHROPIC_ZDR_ENABLED` and uses it only for (a) emitting the correct startup notice and (b) rendering the correct privacy statement in surfaces that expose privacy state. It does not send a ZDR header; it does not filter models by ZDR-eligibility.
- #013's language "sends the appropriate request header when set" and "restrict model selection to ZDR-supported variants" is superseded by this decision; those phrasings are wrong under Anthropic's actual ZDR model.
- The mandatory-startup-notice behavior from #013 point 7 is preserved, but the notice text distinguishes "ZDR attested" from "ZDR enabled" (no flag enables ZDR; only the Anthropic arrangement does).
- User-facing privacy statement updated to include 30-day default retention, 2-year violation retention, 7-year classification retention, the operator-attestation framing, and the explicit HIPAA-not-supported statement. `README.md` ships the updated statement in the same commit as this decision; `docs/deployment.md` will carry it at public flip.
- V1.5+ `LLMProvider` implementations (OpenAIProvider, etc.) must publish their own retention-and-attestation semantics analogous to #015 before they ship; providers without documented retention cannot satisfy the `LLMProvider` Protocol.
- HIPAA-readiness scenarios are explicitly refused by the user-facing privacy statement and deferred to a future decision — a V1 deployment that needs HIPAA has to arrange it separately and will require its own decision entry covering the three gap categories in point 7.

**Referenced from.** `llm/anthropic_provider.py` (when written), `llm/base.py` (when written), `README.md`, `docs/deployment.md`.

---

## 016. LLM exchanges stored locally under retention; logs stay metadata-only

**Status:** Accepted, 2026-04-23. Amends #013 point 5. Extends #014's metadata-vs-content pattern with a new content table.

**Amended 2026-04-26:** point 3's wording corrected. The original "Default TTL matches `findings`" phrasing was internally inconsistent with the plan-level numbers (`llm_call_content` 90d ≤ `findings` 180d). Architectural intent was always "shorter than findings — sensitivity hierarchy, most-sensitive content has shortest TTL"; the wording is now aligned. No change to architectural shape or to any other point in this decision.

**Amended 2026-04-27:** point 1's wording on `event_id` corrected. The original "plain UUID (not a FK with CASCADE)" framing over-applied #014's parent→child dangling-reference pattern in the wrong direction. `llm_call_content.event_id → audit_events.event_id` is child→parent, with the parent append-only forever per #014, so the correct shape is a real FK with `ON DELETE NO ACTION` — which is what the migration shipped and what `docs/schema.md` "Content-table foreign-key semantics" describes. Only the citation drifted. No change to architectural shape or to any other point in this decision.

**Context.** #013 point 5 mandated that prompt and completion content never appear on the audit row, with the rationale "if debugging requires the content, retrieve it by replay from `PRContext` + scope extraction." That guard targeted a threat that doesn't exist under #011: there is no third party gaining access to user content from local database storage. The operator already has the user's code on their own infrastructure; storing the LLM exchange about that code in the same database adds no new exposure surface. The replay story improves materially — within the retention window, every LLM call's full input and output is reconstructable, which is what "every decision is auditable" should actually deliver. The original #013 framing was a SaaS-shaped rule applied to a self-hosted V1; #016 corrects it.

The two surfaces (logs vs database) need different rules: logs flow to stdout, log aggregators, and possibly third-party SIEMs that the operator may not fully control; the database is local and operator-administered. Storing content in the database is acceptable under self-hosted; logging content exposes it to surfaces we don't control. Different surfaces, different rules.

**Decision.**

1. **New content table `llm_call_content`.** Stores prompt and completion text for every LLM call. Keyed by `event_id` (matching the corresponding `LLMCallEvent` audit row). Schema:
   - `event_id UUID PK` — real FK to `audit_events.event_id` with `ON DELETE NO ACTION`. The FK direction is reversed from `audit_events.review_id`: this content row is the child, and the parent audit row is append-only forever per #014. Normal operation never deletes the parent; if a parent-delete path is introduced accidentally, the FK fails loud rather than cascading or nulling content. See `docs/schema.md` "Content-table foreign-key semantics" for the full reasoning.
   - `prompt TEXT NOT NULL` — the full rendered prompt sent to the provider.
   - `completion TEXT NOT NULL` — the full response text.
   - `retention_expires_at TIMESTAMPTZ NOT NULL` — populated at insert per #012's TTL.
   - `installation_id BIGINT NOT NULL` — denormalized for purge scoping (same pattern as `reviews` and `findings` per #014).
   - `created_at TIMESTAMPTZ NOT NULL`.
   - `is_eval BOOLEAN NOT NULL`.
2. **`LLMCallEvent` audit row stays metadata-only per #014.** Unchanged: token counts, model, cost, latency_ms, prompt_hash, cache_hit, context_summary, prompt_template_version, system_prompt_hash, degraded_mode. The audit row records the *fact* of the LLM call and its costed metadata; the content lives separately. The `prompt_hash` on the audit row remains useful even after the content is purged — it lets a replay verify, against the surviving audit metadata alone, that the prompt structure matched what the template would have produced at that policy version.
3. **Retention and purge follow #012 + #014.** `llm_call_content` rows carry `retention_expires_at` populated at insert, are queried by `sweep/purge_expired.py` for expiry-based deletion, and are purged on `installation.deleted` via the grace-window mechanism. `purge_audit` logs the deletion against `llm_call_content` as the target table. **Default TTL is shorter than or equal to the `findings` TTL** — LLM exchange content is more sensitive than finding metadata (carries actual prompt and completion text, including code from PRs), so the most-sensitive content has the shortest TTL. The shape (LLM content TTL ≤ findings TTL) is the architectural anchor; specific numbers live in operator configuration per ITERATION_LOG initial defaults.
4. **Logs stay metadata-only.** The structured logger emits LLMCallEvent's metadata fields only — never the prompt or completion text. Defense in depth: a log filter rejects records containing prompt or completion fields (first line); the logger schema itself omits content fields (belt). The two layers are independently sufficient: if a future code path constructs an ad-hoc log line bypassing the schema, the filter still catches it; if the filter is misconfigured, the schema's omission still prevents it. This rule is unchanged from #013 point 5's original spirit, just narrowed: it applies to log records, not to database storage. The user-facing privacy statement (point 6) names the surface distinction explicitly.
5. **Replay equivalence per #014 point 4 expands.** Full-replay mode (within retention, content present) now reconstructs LLM exchanges with full prompt and completion text from `llm_call_content`, in addition to the existing full-finding reconstruction. Metadata-only-replay mode (post-purge) reconstructs from audit metadata alone — token counts, prompt_hash for structure-verification, system_prompt_hash, context_summary — no content text. The replay tool's mode-distinguishing behavior from #014 applies to LLM content the same way it applies to finding content: the tool refuses to silently produce a hybrid.
6. **User-facing privacy statement (revised from #015 point 5) gains a stored-content clause.** The statement in README and deployment docs becomes: "Outrider stores LLM request and response content in your local database under configured retention TTL (default values in operator configuration; purged on `installation.deleted` along with reviews and findings, per DECISIONS.md #012 + #014). Outrider does not transmit stored LLM content to any third party other than the configured LLM provider at request time per #013/#015's egress rules." Any change to this text — egress phrasing, retention framing, scope of "stored content" — requires a supersession decision, not an in-place README edit.

**Consequences.**

- **Single-transaction insert is required, not optional.** The `LLMCallEvent` audit row insert and the `llm_call_content` row insert happen in a single database transaction. If the transaction fails, neither row exists. This is required for the replay tool's mode distinction (point 5) to work correctly — a missing content row paired with a present audit row would be ambiguous between "purged per retention" (correct mode-distinction signal) and "insert failed" (a third state the dashboard cannot distinguish from the first). Implementations that put the two writes in separate transactions reintroduce that ambiguity and violate this decision.
- New Alembic migration step: create `llm_call_content` table with `retention_expires_at` and `installation_id` columns, indexed for the sweep job's expiry query and the installation-scoped purge query.
- `audit/events.py::LLMCallEvent` schema is unchanged — metadata-only per #014 stands. Content does not move into the audit row.
- Logging filter from #013 point 5 stays in place. The filter is now the *only* defense for logs; the schema-level omission on `LLMCallEvent` doesn't help logs because logs construct their own field set.
- Dashboard's review-detail view renders LLM exchanges within retention. After retention, content is rendered as "content redacted per retention" with the purge date from `purge_audit` — same UX pattern as #014 point 3 for findings.
- Eval scenarios at `tests/eval/scenarios/replay/` cover full-replay (with `llm_call_content` populated) and metadata-only-replay (with `llm_call_content` purged). A retention-boundary scenario (some calls within retention, some past) tests that the replay tool refuses hybrid output and signals which calls are reconstructable in full.
- `README.md` privacy paragraph ships the revised statement (point 6) in the same commit as this decision per the coupling rule established at #015. `docs/deployment.md` carries the same statement at public flip.
- The "guard against third-party content exposure" rationale from #013 point 5 is removed; under #011 it never applied. If V1.5+ adopts SaaS, the LLM-content storage decision is revisited as part of that supersession, alongside everything else under "users own their data" — likely either removing local content storage entirely or adding tenant-scoped encryption-at-rest.

**Referenced from.** `llm/anthropic_provider.py` (when written), `audit/events.py` (when written; LLMCallEvent stays metadata-only), `db/models/llm_call_content.py` (when written; new content table per this decision), `db/models/reviews.py` (when written; same content-table pattern), `README.md`, `docs/deployment.md`, `tests/eval/scenarios/replay/` (when written).

---

## 017. Trace decisions aggregate per `source_finding_id` with full candidate set

**Status:** Accepted, 2026-04-29.

**Amended 2026-04-29 (same day, two clauses):** (a) points 1 + 2 tightened — "per trace round" language dropped in favor of "across the review." A reviewer audit pass minutes after the original landing caught an internal inconsistency: the decision text said "one TraceDecision per source_finding_id **per trace round**" while the reducer key was `source_finding_id` alone, with no `phase_id` / `trace_round_id` on `TraceDecision` or `TraceDecisionEvent`. If a later trace round revisited the same source finding under that text, the reducer would silently drop it. Resolution: commit to the simpler semantic that matches the actual data flow. Findings flow through trace once, in the round in which `analyze` produced them — the trace node looks at NEW findings from the latest analyze pass, not at findings already-considered. So "one TraceDecision per source_finding_id across the review" is what the data flow describes; the "per trace round" qualifier was an author-side leak from thinking about the trace ⇄ analyze loop. If a future feature genuinely needs per-round revisit semantics, it would add `trace_round_id` to the type/event and key on `(trace_round_id, source_finding_id)` via a fresh decision superseding this one. (b) Point 3 clarified: a follow-up audit pass caught a wording ambiguity — the original "unresolved has zero candidates, ambiguous has multiple" framing conflated two different cardinalities (LLM-proposed vs ast_facts-resolved). Per `docs/architecture.md`, the LLM proposes candidates first, ast_facts validates them second; import-resolution failures can leave a non-empty LLM-proposed list with zero ast_facts-validated entries. The corrected framing: `candidates_considered` = the LLM-proposed list (any cardinality); `resolution_status` describes how many of those resolved through ast_facts (zero / exactly one / multiple); `target_file` is the resolved one when `resolved`, None otherwise. The cross-field validator gains a third rule: when `resolved`, `target_file in candidates_considered`. No change to architectural shape; both clauses preserve intent and clarify wording. Same-day amendment marker preserves the history of the corrections for replay-traceability.

**Context.** Two recent canonical-shape corrections within the audit-events module spec drafting surfaced a third question the original spec did not anticipate: with `TraceDecision.target_file` corrected to nullable (None for unresolved/ambiguous decisions per the audit pass that flagged the §4.1.4 line 935 wording), and with `TraceDecision.candidates_considered` corrected to required-no-default (load-bearing for §8.7 replay equivalence), the existing reducer dedup key `(source_finding_id, target_file)` collapses multiple unresolved/ambiguous decisions for the same finding onto `(source_finding_id, None)` and silently swallows them via `append_with_dedup_by`. The data flow per `docs/architecture.md` describes one decision per source finding: the LLM produces one ranked candidate list per finding, the trace node evaluates the list, picks zero or one resolution, and records one outcome. The multi-row-per-finding shape that the reducer key implied is a workaround for a model that does not match the flow. This is genuinely new architectural commitment territory, not a citation-fidelity correction — the original spec did not address what happens when target_file is None for multiple decisions on the same finding, because the original spec did not anticipate the question.

**Decision.** `TraceDecision` aggregates per `source_finding_id`. The reducer dedup key is `source_finding_id` alone (not `(source_finding_id, target_file)`).

1. **One `TraceDecision` per source finding across the review.** The trace node produces exactly one decision per finding it considers: a single `resolution_status` (`resolved` / `unresolved` / `ambiguous`), a single `target_file` (the selected candidate when resolved, `None` otherwise), and the full ranked `candidates_considered` list the LLM proposed at decision time. Multiple candidate paths are represented inside `candidates_considered`, not as multiple `TraceDecision` rows. The trace node sees each finding once — in the round in which `analyze` produced it — so per-round qualifier is unnecessary.
2. **Reducer key is `source_finding_id`.** `append_with_dedup_by(lambda d: d.source_finding_id)` replaces the previous `(source_finding_id, target_file)` key. The reducer dedup-on-key behavior makes replay idempotent: the same trace decision applied twice (webhook redelivery, checkpoint replay, retry) is a no-op. If a future feature genuinely requires re-evaluating the same finding across rounds with potentially different outcomes, that feature must supersede this decision via a fresh DECISIONS entry that adds `trace_round_id` (or equivalent) to both `TraceDecision` and `TraceDecisionEvent` and rekeys the reducer accordingly. Without that fresh decision, the dedup-on-`source_finding_id` rule means a re-evaluated finding's later decision is dropped silently — which is correct under the current commitment (one decision per finding, first wins) and incorrect under a hypothetical future revisit-semantics commitment.
3. **`target_file` is the selected candidate when resolved, None otherwise.** `candidates_considered` is the **LLM-proposed candidate list** (any cardinality — zero, one, or many — depending on what the LLM produced); `resolution_status` describes the outcome of attempting to resolve those candidates through `ast_facts`: `resolved` means exactly one candidate resolved successfully through the import registry; `unresolved` means zero resolved (the LLM may have proposed any number — import-resolution failures, complex/dynamic imports, etc., can produce a non-empty `candidates_considered` with `resolution_status="unresolved"`); `ambiguous` means multiple resolved (the LLM proposed candidates that ast_facts couldn't disambiguate). For `resolution_status == "resolved"`, `target_file` is one of the strings in `candidates_considered` — the LLM-proposed candidate that ast_facts successfully validated. For `unresolved` or `ambiguous`, `target_file is None` (couldn't resolve to a single concrete file). A cross-field `model_validator(mode="after")` enforces three rules: (a) `resolved` ↔ non-None `target_file`; (b) `unresolved` / `ambiguous` ↔ `target_file is None`; (c) when `resolved`, `target_file in candidates_considered` (the resolved selection must be one of the LLM-proposed candidates).
4. **Replay reconstructs the full resolution context.** A reviewer reading the audit log can see, for any finding the trace node touched: the full ranked candidate list the LLM proposed, the outcome the trace node reached, and (when resolved) which candidate became the resolved target. That is the proof artifact for INFERRED findings — INFERRED-tier findings derived from a resolved trace decision are reconstructable end-to-end (proof boundary intact). Unresolved/ambiguous decisions downgrade resulting findings to JUDGED per the existing §4.1.4 comment.
5. **No supersession of prior decisions.** This entry is a fresh commitment; no prior `DECISIONS.md` entry made the multi-row-per-finding claim explicitly. The reducer-key wording in spec.md §7.1 was an implementation sketch that did not anticipate nullable `target_file`. The resolution is a new commitment, not a revision.

**Consequences.**

- `ReviewState.trace_decisions` reducer changes from `(source_finding_id, target_file)` to `source_finding_id` alone. spec.md §7.1 lines 838-843 updated in lockstep.
- `TraceDecision` (`spec.md §4.1.4`) and `TraceDecisionEvent` (`spec.md §8.2`) carry a `model_validator(mode="after")` enforcing three rules: (a) `resolved` ↔ non-None `target_file`; (b) `unresolved` / `ambiguous` ↔ `target_file is None`; (c) when `resolved`, `target_file in candidates_considered` (the resolved selection must be a member of the LLM-proposed candidate list per amended point 3, clause (b)). The validator on the schema-layer `TraceDecision` is the gate at construction time; the audit event's same-shape validator is the gate at emission time; both must hold for replay to reconstruct cleanly.
- The audit-events module spec (`specs/2026-04-29-audit-events-module.md`) consumes the corrected canonical state. Its `TraceDecisionEvent` shape and tests already align with this commitment.
- `agent/nodes/trace.py` (when written) emits exactly one `TraceDecisionEvent` per finding the trace node considered in the round. The emitter constructs the event with the full candidate set the LLM produced, not the per-candidate evaluation steps.
- The architecture.md trace flow narrative is unchanged — it already described one decision per finding. This decision makes the type-layer commitment that matches the narrative.
- Future features that need to record per-candidate evaluation detail (a hypothetical retry mechanism, an A/B candidate-ranking experiment) cannot accumulate multiple `TraceDecision` rows for the same finding under this decision; they would need to either (a) embed the per-candidate detail inside `candidates_considered` or its successor, or (b) supersede this decision explicitly via a fresh `DECISIONS.md` entry.

**Referenced from.** `spec.md` §4.1.4 (TraceDecision), `spec.md` §7.1 (ReviewState.trace_decisions reducer), `spec.md` §8.2 (TraceDecisionEvent), `specs/2026-04-29-audit-events-module.md` (consumer of the corrected canonical state), `agent/nodes/trace.py` (when written), `agent/reducers.py` (when written; carries the dedup-keyed reducer machinery).
