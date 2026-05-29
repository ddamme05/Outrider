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

**Amended 2026-04-30:** the "CI runs the same checks as a secondary gate" framing in the Decision section narrows to "tracked-file-safe subset of pre-commit hooks." Surfaced during the eval-harness spec drafting (the first surface to actually wire up CI per `docs/workflow.md`'s "CI lands the first time a real test is written" rule). The original framing assumed all hook-relevant files would be tracked; under the public/local split established in `docs/workflow.md`, `docs/` is gitignored, so any hook that defaults to reading `docs/spec.md` / `docs/invariants.md` cannot run in CI. Today that's exactly one hook: `invariants-in-sync` (runs `extract_invariants.py --check`, which defaults to those gitignored paths per `scripts/extract_invariants.py:40-41`). Future hooks that depend on gitignored paths join the same excluded list. Architectural intent is unchanged — pre-commit remains the primary gate; CI's secondary role still catches every check that doesn't depend on local-only state. The amendment makes the mechanism honest. No change to the local pre-commit configuration.

**Context.** `scripts/extract_invariants.py --check` detects drift between `docs/spec.md` and `docs/invariants.md`, but the check only has value if it runs before problematic commits land. CI-only enforcement catches drift at PR time — after the author has moved on, after the problematic commit is already in git history, and every failure requires an additional push that pollutes PR history. Local enforcement at commit time catches drift in the same context where it was introduced.

The same reasoning applies to `scripts/test_extract_invariants.py`: the extractor's output is only trustworthy when its tests pass, and a broken extractor edit that silently emits wrong invariants is a worse failure mode than a PR that fails CI.

**Decision.** Ship `.pre-commit-config.yaml` at the repo root with local hooks that use the pinned project virtualenv. Ruff and Ruff format run first so formatting doesn't invalidate extraction:

1. **`ruff`** runs `.venv/bin/ruff check --fix`.
2. **`ruff-format`** runs `.venv/bin/ruff format`.
3. **`invariants-in-sync`** runs `.venv/bin/python scripts/extract_invariants.py --check` and is scoped via `files:` to fire only when `docs/spec.md`, `docs/invariants.md`, or `scripts/extract_invariants.py` changes. Zero overhead on unrelated commits.
4. **`extractor-tests-pass`** runs `.venv/bin/python scripts/test_extract_invariants.py` and is scoped to fire only when the extractor or its test file changes.
5. **`decision-refs-resolve`** runs `.venv/bin/python scripts/check_decision_refs.py` and is scoped to DECISIONS.md or the checker script.

The hooks use `language: system` because the repo already pins the toolchain in `uv.lock` and installs it into `.venv` via `uv sync --dev`. This avoids network fetches during hook execution and avoids relying on a `python` shim that may not exist on every machine.

CI runs the tracked-file-safe subset of these hooks as a secondary gate (per Amended 2026-04-30 above): `ruff`, `ruff-format`, `decision-refs-resolve`, `decision-refs-tests-pass`, `extractor-tests-pass`. `invariants-in-sync` stays local-side only because `extract_invariants.py --check` defaults to gitignored `docs/` paths. If a contributor skips `pre-commit install`, CI catches the drift at PR time for hooks that don't depend on local-only state — this is acceptable as a fallback but not the primary defense.

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

**Amended 2026-05-05:** point 2's metadata-only field set extends with `pricing_version: str` on `LLMCallEvent`. Surfaced during the LLM-provider-wrapper spec drafting (`specs/2026-05-05-llm-provider-wrapper.md` Pending DECISIONS amendment #5). The wrapper computes `cost_usd` provider-side from `llm/pricing.py`'s per-model `Decimal` rate table, with `PRICING_VERSION` as a module constant that bumps when rates change; without recording the version on the event, historical-row replay accuracy depends on maintaining an external version-effective-range map. Recording the version directly on the event matches the `severity-policy-versioned-for-replay` precedent established in #001 — same architectural shape: a deterministic check produces a value; the version it was produced under is recorded on the event row so future replay never depends on an external lookup. Concrete change: `LLMCallEvent` gains `pricing_version: str` as a required field; the wrapper sets it from `PRICING_VERSION` at construction; replay reads it directly instead of inferring from `timestamp`. No change to any other point or to any other event shape; the existing field set is unchanged otherwise.

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
2. **`LLMCallEvent` audit row stays metadata-only per #014.** Unchanged: token counts, model, cost, latency_ms, prompt_hash, cache_hit, context_summary, prompt_template_version, system_prompt_hash, degraded_mode, degradation_reason. The audit row records the *fact* of the LLM call and its costed metadata; the content lives separately. The `prompt_hash` on the audit row remains useful even after the content is purged — it lets a replay verify, against the surviving audit metadata alone, that the prompt structure matched what the template would have produced at that policy version. (The `degradation_reason: Literal["parse_failed", "tree_has_error_in_changed_regions"] | None` field was added in the analyze-foundation arc per `specs/2026-05-19-analyze-foundation.md` §0b: the prior shape carried `degraded_mode: bool` alone, which loses the typed cause on metadata-only replay — three-agent convergent finding from the §0b crazy-audit; folded by extending the schema rather than the spec because the omission would otherwise corrupt replay reconstruction. Same metadata-only category as `degraded_mode` and `prompt_template_version`.)
3. **Retention and purge follow #012 + #014.** `llm_call_content` rows carry `retention_expires_at` populated at insert, are queried by `sweep/purge_expired.py` for expiry-based deletion, and are purged on `installation.deleted` via the grace-window mechanism. `purge_audit` logs the deletion against `llm_call_content` as the target table. **Default TTL is shorter than or equal to the `findings` TTL** — LLM exchange content is more sensitive than finding metadata (carries actual prompt and completion text, including code from PRs), so the most-sensitive content has the shortest TTL. The shape (LLM content TTL ≤ findings TTL) is the architectural anchor; specific numbers live in operator configuration per ITERATION_LOG initial defaults.
4. **Logs stay metadata-only.** The structured logger emits LLMCallEvent's metadata fields only — never the prompt or completion text. Defense in depth: a log filter rejects records containing prompt or completion fields (first line); the logger schema itself omits content fields (belt). The two layers are independently sufficient: if a future code path constructs an ad-hoc log line bypassing the schema, the filter still catches it; if the filter is misconfigured, the schema's omission still prevents it. This rule is unchanged from #013 point 5's original spirit, just narrowed: it applies to log records, not to database storage. The user-facing privacy statement (point 6) names the surface distinction explicitly.
5. **Replay equivalence per #014 point 4 expands.** Full-replay mode (within retention, content present) now reconstructs LLM exchanges with full prompt and completion text from `llm_call_content`, in addition to the existing full-finding reconstruction. Metadata-only-replay mode (post-purge) reconstructs from audit metadata alone — token counts, prompt_hash for structure-verification, system_prompt_hash, context_summary — no content text. The replay tool's mode-distinguishing behavior from #014 applies to LLM content the same way it applies to finding content: the tool refuses to silently produce a hybrid.
6. **User-facing privacy statement (revised from #015 point 5) gains a stored-content clause.** The statement in README and deployment docs becomes: "Outrider stores LLM request and response content in your local database under configured retention TTL (default values in operator configuration; purged on `installation.deleted` along with reviews and findings, per DECISIONS.md #012 + #014). Outrider does not transmit stored LLM content to any third party other than the configured LLM provider at request time per #013/#015's egress rules." Any change to this text — egress phrasing, retention framing, scope of "stored content" — requires a supersession decision, not an in-place README edit.

**Consequences.**

- **Single-transaction insert is required, not optional.** The `LLMCallEvent` audit row insert and the `llm_call_content` row insert happen in a single database transaction. If the transaction fails, neither row exists. This is required for the replay tool's mode distinction (point 5) to work correctly — a missing content row paired with a present audit row would be ambiguous between "purged per retention" (correct mode-distinction signal) and "insert failed" (a third state the dashboard cannot distinguish from the first). Implementations that put the two writes in separate transactions reintroduce that ambiguity and violate this decision.
- New Alembic migration step: create `llm_call_content` table with `retention_expires_at` and `installation_id` columns, indexed for the sweep job's expiry query and the installation-scoped purge query.
- `audit/events.py::LLMCallEvent` remains metadata-only per #014; content does not move into the audit row. (The 2026-05-05 amendment to point 2 added `pricing_version: str` as a metadata field — same metadata-only category as `prompt_template_version`, recording which versioned table produced the cost. The metadata-only invariant is unaffected.)
- Logging filter from #013 point 5 stays in place. The filter is now the *only* defense for logs; the schema-level omission on `LLMCallEvent` doesn't help logs because logs construct their own field set.
- Dashboard's review-detail view renders LLM exchanges within retention. After retention, content is rendered as "content redacted per retention" with the purge date from `purge_audit` — same UX pattern as #014 point 3 for findings.
- Eval scenarios at `tests/eval/scenarios/replay/` cover full-replay (with `llm_call_content` populated) and metadata-only-replay (with `llm_call_content` purged). A retention-boundary scenario (some calls within retention, some past) tests that the replay tool refuses hybrid output and signals which calls are reconstructable in full.
- `README.md` privacy paragraph ships the revised statement (point 6) in the same commit as this decision per the coupling rule established at #015. `docs/deployment.md` carries the same statement at public flip.
- The "guard against third-party content exposure" rationale from #013 point 5 is removed; under #011 it never applied. If V1.5+ adopts SaaS, the LLM-content storage decision is revisited as part of that supersession, alongside everything else under "users own their data" — likely either removing local content storage entirely or adding tenant-scoped encryption-at-rest.

**Referenced from.** `llm/anthropic_provider.py`, `audit/events.py` (`LLMCallEvent` stays metadata-only), `audit/persister.py` (atomic `LLMCallEvent` + `llm_call_content` INSERT per this decision; back-pointer comments at `persister.py:20+`), `audit/sinks.py` (sink Protocols consumed by the persister), `db/models/llm_call_content.py` (when written; new content table per this decision), `db/models/reviews.py` (when written; same content-table pattern), `README.md`, `docs/deployment.md`, `tests/eval/scenarios/replay/` (when written).

---

## 017. Trace decisions aggregate per `source_finding_id` with full candidate set

**Status:** Accepted, 2026-04-29. Amended 2026-05-24 by #024 (field rename + new field + validator rewrite to accommodate import-string candidate shape; see #024 for the full delta).

**Amended 2026-05-24 by #024 (event/state field shape):** #024 commits trace's V1 candidate shape to dotted Python import strings (not file paths). The renaming-and-splitting required to keep #017's once-per-finding validators coherent under import-string semantics is carried inside #024 and summarized here for the chain: `candidates_considered: tuple[str, ...]` is renamed to `proposed_import_strings: tuple[str, ...]` (the LLM-proposed dotted forms); a new field `resolved_candidate_paths: tuple[str, ...]` carries the resolver outputs from `coordinates.resolve_candidate_paths`. The three cross-field validator rules of point 3 below are rewritten to consult `resolved_candidate_paths` instead of `candidates_considered`: `resolved` → `len(resolved_candidate_paths) == 1` AND `target_file == resolved_candidate_paths[0]`; `unresolved` → `len(resolved_candidate_paths) == 0` AND `target_file is None`; `ambiguous` → `len(resolved_candidate_paths) > 1` AND `target_file is None`. The uniqueness validator splits into two — one per tuple. `target_file`, when non-None, also passes through `validate_diff_path` at the audit-shadow boundary (defense-in-depth at the append-only log). The core commitment of #017 — one decision per `source_finding_id` across the review, reducer key on `source_finding_id` alone — is unchanged. The amendment is a wire-shape rewrite, not a semantic supersession.

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

**Post-#024 reading (added 2026-05-24 to reconcile the body with the amended field shape).** The original Decision text above commits the shapes that held as of 2026-04-29: `TraceDecision.candidates_considered`, the `target_file in candidates_considered` validator, and `target_file` as the optional single-string resolution outcome. #024 (Accepted 2026-05-24) renames `candidates_considered → proposed_import_strings` (the LLM-proposed dotted forms) and adds `resolved_candidate_paths` (the resolver outputs) on BOTH `TraceDecision` and `TraceDecisionEvent`. The three cross-field validator rules (point 3 above) consult `resolved_candidate_paths`, not `candidates_considered`, for the resolved/unresolved/ambiguous cardinality and the `target_file == resolved_candidate_paths[0]` membership check. The uniqueness validator splits into two — one per tuple. The "audit-events module spec consumes the corrected canonical state" Consequence above pre-dates #024; the audit-events module shipped with the original field names and was re-amended in lockstep with #024. The core commitment of #017 — one `TraceDecision` per `source_finding_id` across the review, reducer key `append_with_dedup_by(lambda d: d.source_finding_id)` — stands unchanged. When in doubt about field names today, read #024; when in doubt about the cardinality contract, read this entry.

**Referenced from.** `spec.md` §4.1.4 (TraceDecision), `spec.md` §7.1 (ReviewState.trace_decisions reducer), `spec.md` §8.2 (TraceDecisionEvent), `specs/2026-04-29-audit-events-module.md` (consumer of the corrected canonical state), `agent/nodes/trace.py` (when written), `agent/reducers.py` (when written; carries the dedup-keyed reducer machinery).

## 018. `FileExaminationEvent` carries `skip_reason` for skipped files

**Status:** Accepted, 2026-05-01.

**Context.** Drafting the V1 ast_facts/ implementation spec (`specs/2026-04-30-ast-facts-module.md`) surfaced a contradiction within canonical between §5.5 ("the audit log records a file_skipped event with the reason") and §8.2 (which enumerates `FileExaminationEvent` fields as `file_path`, `examination_type`, `node_id`, `parse_status` — no reason field, and no separate `FileSkippedEvent` type). The contradiction predates this feature work; ast_facts/ ships `ParseResult.skip_reason: SkipReason | None` as in-process metadata regardless, but whether and how that reason propagates to the audit event is owned by canonical, not by a feature spec. This entry resolves the canonical contradiction so the V1 implementation spec can transition to Approved.

**Decision.** `FileExaminationEvent` gains a nullable `skip_reason: SkipReason | None` field, populated by the consuming node when emitting the event for a skipped file. No new event type; no discriminator migration.

1. **Field shape.** `FileExaminationEvent.skip_reason: SkipReason | None = None`. Non-`None` only when `parse_status == "skipped"`. The five `SkipReason` enum values (`OVERSIZED`, `VENDORED`, `GENERATED_FILENAME`, `MINIFIED`, `GENERATED_BANNER`) are defined in `ast_facts/models.py` per the V1 ast_facts/ spec; the audit schema imports rather than redefines them.

2. **Cross-field validator.** A `model_validator(mode="after")` on `FileExaminationEvent` enforces `skip_reason is not None` ↔ `parse_status == "skipped"`. The other three `parse_status` values (`clean`, `degraded`, `failed`) require `skip_reason is None`. Same shape as `TraceDecisionEvent`'s `(target_file, resolution_status)` validator per #017 — one event, one related-but-nullable field, one cross-rule, deterministic on replay.

3. **A separate `FileSkippedEvent` was rejected.** Two event types per file lifecycle (or branch-on-emission to pick one) introduces replay-side complexity no other Outrider event has. The single-event-with-nullable-field shape matches #017's design and keeps the file-examination lifecycle in one event.

4. **Amending §5.5 to drop the reason claim was also rejected.** The "every skipped file is auditable with its reason" property is what makes the planned dashboard skip-panel actionable. Honoring §5.5 via §8.2 extension is cheaper than walking back §5.5's promise.

5. **Migration scope: payload-schema only, no Alembic.** Audit events serialize Pydantic models into the `audit_events` JSONB payload column. Adding `skip_reason` is a Pydantic field addition with a default of `None`. No Alembic migration is required (no column add, no JSON-schema constraint to update at the DB level — Outrider's audit-payload pattern per #014 / #016 doesn't use DB-level JSON constraints). **No backfill is required because no historical `FileExaminationEvent` rows exist:** V1 has not shipped and the audit_events table is empty in every deployed environment. (`audit/events.py` already exists as Status:Complete per the audit-events module spec; the `skip_reason` field addition is sequenced after `outrider/ast_facts/models.py` lands so the `SkipReason` import is satisfiable.) Note that the cross-field validator at point 2 *would* reject a historical skipped payload missing the new key (default-`None` collides with `parse_status="skipped"`); this would matter if such payloads existed, but they do not. Future schema changes that introduce a similar interactive cross-field constraint *after* audit data exists must consider a backfill or compatibility shim — out of scope for this entry, named here so the pattern is on the record for future authors.

6. **`SkipReason` enum ownership and import boundary.** `SkipReason` lives in `ast_facts/models.py` (where `should_skip` is implemented and where new rules will be added in V1.5+ for JS/TS). `audit/events.py` imports it via `from outrider.ast_facts.models import SkipReason`. **`outrider/ast_facts/__init__.py` must stay import-light** so this import does not transitively load `python_adapter.py` (which imports `tree_sitter`) into audit's module graph: the package `__init__.py` re-exports light types (Pydantic models, Protocols, errors, `SkipReason`) eagerly but uses a module-level `__getattr__` for `parse_python`, lazy-loading `python_adapter.py` only when `parse_python` is accessed. This keeps `from outrider.ast_facts.models import ...` cheap for audit/ and other consumers that don't need the adapter, and matches how `audit/replay.py` already imports `QueryMatchSpan` from `ast_facts/models.py` per the V1 ast_facts/ spec. The dependency edge stays one-way: `audit → ast_facts` (no cycle).

**Consequences.**

- `spec.md` §8.2's `FileExaminationEvent` field list extends to `file_path, examination_type, node_id, parse_status, skip_reason`. Documentation update to §8.2's table is reflected in local `docs/spec.md`.
- `spec.md` §5.5 wording is amended locally to read `FileExaminationEvent` rather than `file_skipped event` — a wording fix, not a behavioral change.
- `spec.md` §5.5's exclusion-pattern enumeration is also amended to include the oversized rule alongside vendored / generated / minified (the V1 ast_facts/ spec adds `SkipReason.OVERSIZED`/`MAX_PARSE_BYTES` as a fifth exclusion category; §5.5 currently lists only the original three categories from before the DoS-posture work).
- `audit/events.py` adds `skip_reason: SkipReason | None = None` to `FileExaminationEvent` plus the cross-field validator. Imports `SkipReason` from `ast_facts.models`. (`audit/events.py` already exists as Status:Complete per `specs/2026-04-29-audit-events-module.md`; the field addition is sequenced after `outrider/ast_facts/models.py` lands so the `SkipReason` import is satisfiable.)
- `audit/replay.py` (when written) replay is unchanged for clean/failed/degraded events; for skipped events, replay can additionally assert the stored `skip_reason` is one of the five canonical enum values.
- `ast_facts/__init__.py` uses module-level `__getattr__` for `parse_python` to keep light-type imports (`SkipReason`, models, Protocols) cheap for callers that don't need the tree-sitter-loading adapter. Import-light is a hard property the V1 ast_facts/ spec must enforce; **a subprocess-based regression test catches reversions** (lives at `tests/integration/test_ast_facts_query_registry.py::test_import_light_subprocess_isolated`): `subprocess.run([sys.executable, "-c", 'import sys; from outrider.ast_facts.models import SkipReason; assert "tree_sitter" not in sys.modules, sorted(sys.modules)'])` must exit 0. The subprocess isolation is load-bearing — running the assertion in the shared pytest interpreter would fail spuriously if any earlier test legitimately imported `tree_sitter`, masking real reversions and producing false positives. The fresh-interpreter form measures the actual import graph of `outrider.ast_facts.models` in isolation.
- `specs/2026-04-30-ast-facts-module.md` Approval prerequisite is satisfied; the spec has transitioned from Drafted to Approved (2026-05-01) on this entry's landing. The consuming node (analyze / intake) populates `FileExaminationEvent.skip_reason` from `ParseResult.skip_reason` directly when emitting the event.
- Dashboard rendering of the matched skip rule is unblocked: the dashboard reads `FileExaminationEvent.skip_reason` from the audit event stream and renders the matched rule per file. The deferred-rendering language in the ast_facts/ V1 spec is replaced by a normal dependency on this entry.
- Eval-harness inspection still reads `ParseResult.skip_reason` directly for in-test assertions; nothing about that path changes. Structured INFO logging similarly continues to read from `ParseResult`.
- No Alembic migration. Pydantic payload-schema extension is the entire audit-side change.
- V1.5+ adds new `SkipReason` enum values (e.g., JS/TS-specific patterns) in `ast_facts/models.py`; `FileExaminationEvent.skip_reason` accepts them automatically. No cascading audit-schema change per new rule.

**Referenced from.** `spec.md` §5.5 (audit-records-the-reason wording, exclusion-pattern enumeration), `spec.md` §8.2 (`FileExaminationEvent` field list), `specs/2026-04-30-ast-facts-module.md` (Approval prerequisite, Audit Events Emitted note), `src/outrider/audit/events.py` (`FileExaminationEvent.skip_reason` field + `_enforce_skip_reason_outcome` validator + `SkipReason` import), `src/outrider/ast_facts/__init__.py` (module-level `__getattr__` lazy-load of `parse_python` + subprocess-isolated import-light regression test in `tests/integration/test_ast_facts_query_registry.py`), `src/outrider/ast_facts/models.py` (`SkipReason` enum + `ParseResult.skip_reason` field), `audit/replay.py` (when written — replay can additionally assert stored `skip_reason` is one of the canonical enum values).

**Amended 2026-05-20** — three new `SkipReason` enum values for analyze-stage skip causes.

The original five `SkipReason` values cover the ast_facts/-stage exclusion rules (`OVERSIZED`, `VENDORED`, `GENERATED_FILENAME`, `MINIFIED`, `GENERATED_BANNER`) — all decisions the parser can make before the analyze node runs. Analyze itself can also skip a file for reasons the parser cannot: the cost budget is exhausted before analyze reaches it, the file has no reviewable diff context after parse-failure (binary or pure-deletion), or the file's changes don't intersect any scope unit (comment-only, whitespace-only, module-level). Each case needs its own `SkipReason` value so the audit row distinguishes them and dashboard skip-reason aggregates render meaningfully.

Adds three values to `ast_facts/models.py::SkipReason`:

- `COST_BUDGET_EXHAUSTED` — pre-flight budget gate in the analyze node refused to run an LLM call against this file because the cumulative per-review cost has already crossed the configured ceiling.
- `NO_REVIEWABLE_CONTEXT` — parse failed AND the diff carries no addable hunks (binary file, pure deletion) — there's nothing for even a JUDGED-tier degraded-mode finding to anchor against per the §4 `span_within_degraded_context` admission rule.
- `NO_CHANGED_SCOPE_UNITS` — file parsed cleanly but the diff hunks don't intersect any scope unit (comment-only, whitespace-only, module-level-only changes). Sending such a file through analyze would consume budget for no value.

The `SkipReason` Literal in `FileExaminationEvent` accepts the new values automatically because it inherits from the enum. No DB migration (per the original #018 point 5: `audit_events.payload` is JSONB; no schema constraint). No backfill (audit log carries no `FileExaminationEvent` rows from analyze yet — analyze hasn't shipped).

**Consumer:** the sister `specs/YYYY-MM-DD-analyze-implementation.md` spec's analyze-node body sets one of these three values on `FileExaminationEvent.skip_reason` when it skips a file mid-pass. The foundation spec (`specs/2026-05-19-analyze-foundation.md` §0a) lands the enum additions in `src/outrider/ast_facts/models.py` so they're available before the analyze-implementation spec consumes them.

**Naming-axis trade-off (decided 2026-05-20).** The original #018's five values follow a naming convention rooted in the file's content (`OVERSIZED`, `VENDORED`, etc.). The three new values name analyze's decision rationale (`COST_BUDGET_EXHAUSTED`, `NO_REVIEWABLE_CONTEXT`, `NO_CHANGED_SCOPE_UNITS`) — a different naming axis. The mixed-axis cost was weighed against the alternative — a single `SKIPPED_BY_ANALYZE` value with a separate `skip_detail: str` field — and rejected because (a) the three reasons need to be enumerable for dashboard aggregation, and (b) free-text detail would defeat #014's structural-metadata-only audit rule. The mixed naming axis is the accepted cost. Downstream consumers that need to discriminate the two axes use `SkipReason.stage() -> Literal["parser", "analyze"]` (added in foundation-wide sharp-edges fold I-3); the helper lives on the enum class itself so consumers don't string-parse value names.

**Amended 2026-05-21** — two new `SkipReason` enum values for producer-decision skip causes that the analyze arc has been routing through `OVERSIZED` as a temporary mapping.

Adds two values to `ast_facts/models.py::SkipReason`:

- `BINARY` — content rejected as non-reviewable text by the **intake decode gate**: NUL-byte present OR UTF-8 strict-decode failure. Defined by the producer decision (intake's `_classify_or_reserve_decode` refused to route the bytes to downstream nodes), not by a content-ontology claim that the file is "binary" in any universal sense — a corrupted UTF-8 text file is admitted to this category for the same reason GitHub-API-binary files are.
- `UNSUPPORTED_LANGUAGE` — file path's language is **unsupported by the current analyze implementation**. V1 ships only the Python adapter; `agent/nodes/analyze.py::_process_one_file` admits only `.py` / `.pyi`. Capability-scoped: the value names "today's analyze implementation cannot review this," not "Outrider forever cannot." When V1.5+ adds JS/TS/Go adapters, the analyze gate widens and producer call sites stop using this value; the enum value stays in the catalog for replay-equivalence reconstruction of historical events.

Until 2026-05-21, both cases routed through `SkipReason.OVERSIZED` (per FUP-033's temporary mapping). That alias is no longer tolerable now that producers are shipped enough to appear in audit review — `OVERSIZED` misstates the cause for both binary-decode rejections and non-Python admissions. The 2026-05-20 mixed-axis trade-off accepted "decision rationale" naming for analyze-stage skips; the same axis applies here. `SkipReason.stage()` maps the two new values:

- `BINARY → "parser"` — intake-stage producer decision; `_classify_or_reserve_decode` runs before any parser/analyze admission and shares the bucket with the original five parser-stage values.
- `UNSUPPORTED_LANGUAGE → "analyze"` — the Python-only `_is_python_file` gate inside the analyze node makes the decision.

The `FileExaminationEvent.skip_reason` Literal accepts the two new values automatically because it inherits from the enum. No Alembic migration (per the original #018 point 5: `audit_events.payload` is JSONB; no schema constraint). No production backfill is required because these producer routes have not shipped — the `audit_events` table contains no `FileExaminationEvent` rows that used the temporary `OVERSIZED` routing.

**Consumer side (lands atomically with this amendment):**

- `ast_facts/models.py::SkipReason` adds `BINARY = "BINARY"` and `UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"`. The `_PARSER_STAGE_SKIP_REASONS` and `_ANALYZE_STAGE_SKIP_REASONS` sets receive the new values; the import-time totality + disjointness assertion proves every value lives in exactly one stage.
- `agent/nodes/intake.py::_classify_or_reserve_decode` switches from `SkipReason.OVERSIZED` to `SkipReason.BINARY` on the NUL-byte branch and the `UnicodeDecodeError` branch. Docstring updated to drop the FUP-033 temporary-mapping language.
- `agent/nodes/analyze.py::_process_one_file` switches the non-`_is_python_file(path)` early-return from `SkipReason.OVERSIZED` to `SkipReason.UNSUPPORTED_LANGUAGE`. Inline comment + module docstring updated to drop the FUP-033 temporary-mapping language.
- Tests at `tests/unit/test_intake_node.py` (the `_classify_or_reserve_decode` regression tests) and `tests/unit/test_analyze_node.py::test_non_python_file_routed_to_skip_without_calling_provider` assert the specific new values instead of the temporary `OVERSIZED`.

---

## 019. `schemas/` is for owner-less cross-boundary models; owned protocol/event surfaces live with their owner

**Status:** Accepted, 2026-05-05.

**Context.** The `docs/conventions.md` "File organization" rule says: *"`schemas/` holds only Pydantic models that cross subsystem boundaries."* Read literally, this puts every cross-boundary Pydantic model in `schemas/` — including audit events (consumed by `audit/`, `dashboard/`, `anomaly/`, `replay/`, plus produced by `agent/`) and LLM provider call surfaces (`LLMRequest`/`LLMResponse`/`LLMMessage`, produced by `agent/` nodes and consumed by `llm/`). The audit-events module shipped with `LLMCallEvent` and the rest of the V1 audit event hierarchy in `audit/events.py`, not `schemas/` — establishing a pattern. The LLM-provider-wrapper spec then drafted `LLMRequest`/`LLMResponse`/`LLMMessage` in `llm/base.py` for the same reason. Both placements feel right but technically violate the literal convention. Repeated review cycles re-raise "shouldn't this be in `schemas/`?" — burning attention without changing the answer.

The unstated convention the audit-events precedent established is: **a Pydantic model lives in the module that owns its lifecycle**. `LLMCallEvent` is owned by the audit subsystem (defined and validated by `audit/events.py`; consumers only import + read). `LLMRequest`/`LLMResponse`/`LLMMessage` are owned by the LLM-wrapper subsystem (defined and validated by `llm/base.py`; agent-node callers construct + pass; `llm/` consumes). Owned types stay with their owner; cross-boundary imports are routine and fine.

`schemas/` keeps a real role: Pydantic models that genuinely cross subsystems with **no clear owner** — for example, `PRContext` (built by webhook handler, consumed by every node, audit, dashboard) has no single subsystem that "owns" it; it lives in `schemas/`. That's the case `schemas/` is for.

**Decision.**

1. **`schemas/` holds only Pydantic models with no clear owning subsystem.** Cross-boundary use alone does NOT require `schemas/` placement; it's a necessary but not sufficient condition.
2. **Pydantic models with a clear owning subsystem live with their owner.** "Owning subsystem" means the subsystem that defines the model's invariants, runs its validators, and is the canonical reader for round-trip semantics — not just the producer of instances.
3. **Cross-boundary imports of owned models are explicitly endorsed.** No code-organization gymnastics required (re-exports, intermediate facades, etc.) just to satisfy the literal letter of the previous wording. `from outrider.audit.events import LLMCallEvent` from `agent/nodes/analyze.py` is fine. `from outrider.llm import LLMRequest` from `agent/nodes/triage.py` is fine.
4. **Existing precedent stays.** `LLMCallEvent` in `audit/events.py` (per audit-events module spec) and `LLMRequest`/`LLMResponse`/`LLMMessage` in `llm/base.py` (per LLM-provider-wrapper spec) both stay where they are. No file moves; this is a documentation-of-already-established-pattern entry.
5. **`docs/conventions.md` "File organization"** is amended in the same working change (locally — `docs/` is currently gitignored per the workflow's local-vs-tracked split, so the amendment lives in the local working copy until `docs/` graduates to tracked per `FOLLOWUPS.md` FUP-004's exit rule). The amended wording: "`schemas/` holds Pydantic models that cross subsystem boundaries AND have no single owning subsystem; models with a clear owner live with their owner and are imported across subsystems as needed (precedent: `LLMCallEvent` in `audit/events.py`; `LLMRequest`/`LLMResponse`/`LLMMessage` in `llm/base.py`)." Once `docs/` is tracked, the conventions.md amendment ships in a follow-on commit; the inline-tag pipeline then carries the rule into `docs/spec.md` per the standard invariants-extraction flow.

**Consequences.**

- The literal convention narrows; the de facto convention is now documented and citable.
- Future feature specs no longer relitigate the schema-location question for owned Pydantic surfaces.
- `schemas/` does not become a god-folder of every cross-boundary model; it stays small and reserved for models without an owner.
- New audit event types continue to land in `audit/events.py`. New provider call-surface types continue to land in `llm/base.py`. New non-owned cross-boundary models still go in `schemas/`.

**Referenced from.** `docs/conventions.md` "File organization" (amended in the local working copy alongside this entry; `docs/` is currently gitignored, so the conventions amendment ships publicly when `docs/` graduates per `FOLLOWUPS.md` FUP-004), `src/outrider/audit/events.py` (existing precedent: `LLMCallEvent` and the V1 audit event hierarchy), `src/outrider/llm/base.py` (`LLMRequest` / `LLMResponse` / `LLMMessage` per `specs/2026-05-05-llm-provider-wrapper.md`), `src/outrider/schemas/*.py` (existing models without an owning subsystem stay; e.g., `PRContext`).

## 020. Webhook receiver constructs seed PRContext; intake enriches

**Status:** Accepted, 2026-05-08.

**Context.** Both `docs/spec.md` (§3.3 / §4.1.1) and `docs/architecture.md` carried internally-contradictory framings of who produces `PRContext`: spec.md said intake "outputs" PRContext, while architecture.md said the webhook handler "translates into PRContext." Spec §15.2's `build_graph` snippet calls `state.pr_context.installation_id` inside intake's body, which only works if `PRContext` exists at intake start — making "intake outputs PRContext from nothing" structurally impossible. The contradiction surfaced during the schema-foundation work as Round 7 of an external-reviewer audit cycle: the question "who produces PRContext?" cascaded to `installation_id` placement, eval factory shape, and `ReviewState` seed semantics. Multiple valid resolutions existed (intake produces; webhook produces; two-schema split); without explicit decision, future node specs would have copied whichever framing they happened to read first.

**Decision.** The webhook receiver constructs the seed `PRContext` after signature verification and idempotency checks. The seed contains `installation_id`, repo coordinates (`owner`/`repo`), PR identity (`pr_number`/`base_sha`/`head_sha`), `pr_title`/`author`, optional `pr_body` (None for PRs without a description; GitHub's `pull_request.body` is `string | null`), `total_additions`/`total_deletions`, and an empty `changed_files=()`. The intake node enriches this seed by fetching the file list via `GET /repos/{owner}/{repo}/pulls/{number}/files` plus per-file base/head content via the contents API, constructs a fresh `tuple[ChangedFile, ...]` populated per the post-intake contract — required `path` / `status` / `additions` / `deletions` on every instance; status-aware `content_base` / `content_head` (the side-that-doesn't-exist is None per the §7.2 invariants); status-aware `previous_path` (str for renames, read from GitHub's `previous_filename` field; None otherwise); and best-effort `patch` (None when GitHub's `/pulls/{number}/files` omits `patch` for binary diffs or diffs too large for the API) and `language` (None when intake can't detect a language for the file's extension) — and returns `{"pr_context": new_pr_context}` for the LangGraph reducer to merge into state.

**Consequences.**

- `ReviewState.pr_context: PRContext` stays required at graph start (no `Optional` wrapper); the seed is always populated by the webhook receiver before any node runs.
- `installation_id` lives on `PRContext` (matches §15.2 build_graph snippet's `state.pr_context.installation_id` access pattern).
- `PRContext.installation_id` is plain `int` (no `Field(ge=1)`) per the eval-isolation convention — eval factories use synthetic non-colliding IDs that may be negative; production webhook validation enforces real GitHub IDs at the input boundary, not at the carrier-schema level.
- The webhook seed may use `changed_files=()` because GitHub `pull_request` webhooks do not include the full per-file list; intake replaces it with the fetched `ChangedFile` tuple. `ChangedFile.patch` is `Optional[str]` on the carrier (Round 22 fix; pre-amendment incorrectly claimed every status produces a patch). GitHub's `/pulls/{number}/files` API omits the `patch` property for binary files and for diffs too large for the API to include — applies to ANY status. Downstream code that consumes `patch` must treat `None` as "no textual diff available from GitHub" — not as an empty unified diff with zero hunks. `content_base` and `content_head` are `Optional[str]` on the carrier with status-aware completeness enforced by the validator: the side that doesn't exist for a given status is required to be `None` (e.g., `added` requires `content_base is None`; `removed` requires `content_head is None`), and the side that does exist is required to be a non-None str. `ChangedFile` instances are only ever constructed by intake, after their data is fetched, so the post-intake validator's status-aware checks act as a defense-in-depth gate against malformed upstream / fixture data slipping past the carrier's Optional typing. **Status-aware completeness invariants (Amended 2026-05-08, Round 14):** the post-intake `ChangedFile` schema enforces `added` → `content_head` set / `content_base` None; `removed` → `content_base` set / `content_head` None; `modified`/`renamed` → both content sides set. **`previous_path` for renames (Amended 2026-05-08, Round 14):** `ChangedFile` carries a `previous_path: str | None` field; required str for `status="renamed"` (intake reads from GitHub's `previous_filename` field on the `/pulls/{number}/files` response), None for all other statuses (the validator enforces both directions). **Count-status invariants (Amended 2026-05-08, Round 15):** `added` requires `deletions == 0`; `removed` requires `additions == 0`. GitHub's API can never produce these impossible shapes; the schema doesn't admit them either, so a buggy fixture or upstream can't silently slip a malformed instance past the post-intake contract.
- Webhook-receiver spec (separate, future) owns the seed-construction logic, including signature verification, idempotency check, payload-to-PRContext translation, and ReviewState seed dispatch.
- Intake-node spec (separate, future) owns enrichment logic. Sequencing per spec §4.1.1: intake first calls `GET /repos/{owner}/{repo}/pulls/{number}/files` (sequential — the file list must return before per-file content fetches can be addressed), then dispatches per-file content fetches in parallel via `asyncio.gather`. Path selection depends on `status`: `added` fetches head only at `path`; `removed` fetches base only at `path`; `modified` fetches base + head at the same `path`; `renamed` fetches base at `previous_path` (the pre-rename path; read from GitHub's `previous_filename` field) and head at `path` — the schema's same-path-rename rejection (`previous_path != path`) ensures intake never fans out two fetches against an identical path for a non-rename. PR metadata is NOT in the fetch set — it is webhook-seeded per the producer-boundary call above.
- **Dispatcher carries the seed `ReviewState` (Amended 2026-05-08).** `ReviewDispatcher.dispatch(state: ReviewState) -> None`, not `dispatch(review_id: UUID)`. Pre-amendment §9.2 + §6.5 had the dispatcher carrying only `review_id`, which left no specified path for the seed `PRContext` to reach graph start (no seed-payload storage layer, no graph-start hook to load it). The dispatcher-carries-state path matches LangGraph's state model — state IS the graph input — and works identically for V1 (`background_tasks.add_task(run_graph, state)`) and V2 (`run_graph_task.delay(state.model_dump_json())` round-tripping JSON through the Celery broker). The alternative (`run_graph(review_id)` loads seed from durable storage) was rejected for adding a storage layer no other component uses.

**Referenced from.** `docs/spec.md` §3.3 (line 167 lifecycle paragraph, amended same-pass), `docs/spec.md` §4.1.1 (intake node responsibility, amended same-pass), `docs/spec.md` §6.5 (idempotency-check sample dispatch, amended Round 10), `docs/spec.md` §7.2 (`PRContext.installation_id` field comment + `ChangedFile` invariants comment block), `docs/spec.md` §9.2 (`ReviewDispatcher` Protocol + V1/V2 implementations, amended Round 10), `docs/architecture.md` (intake node responsibility + data-flow lines + request-lifecycle ordering, amended same-pass; ships publicly when `docs/` graduates per `FOLLOWUPS.md` FUP-004), `src/outrider/schemas/pr_context.py` (module docstring + `installation_id` field + `enforce_status_invariants` validator covering R14/R15/R16 status-conditional invariants), `src/outrider/schemas/review_state.py` (skeletal `ReviewState` with webhook-seeded slots + `validate_assignment=True` per R5; ReviewState.pr_context required at graph start per #020), `tests/unit/test_pr_context.py` (status-content + status-count + rename path-shape tests pinning the post-intake contract per R14/R15/R16/R18; `patch=None` four-status admittance + JSON round-trip and `pr_body=None` admittance per R22), `tests/unit/test_review_state.py` (`validate_assignment` guards + reducer-merge dict-coercion semantics per R5/R19), `specs/2026-05-08-schema-foundation.md` (Actual outcome → producer-boundary fold).

---

## 021. `FINDING_TYPE_TO_DIMENSION` is append-only for existing `FindingType` members

**Status:** Accepted, 2026-05-20.

**Context.** The analyze-foundation arc added `policy/dimensions.py::FINDING_TYPE_TO_DIMENSION` (per spec §6) and a `ReviewFinding._enforce_dimension_lockstep` model validator (post-foundation data-integrity audit F2) that asserts `dimension == FINDING_TYPE_TO_DIMENSION[finding_type]` at every construction AND replay-time `model_validate`. The validator closes a real gap — a stored audit_events.payload row with a stale dimension would otherwise survive replay reconstruction under an old mapping — but it also implicitly commits the mapping to immutability: if the mapping ever changes, old audit rows fail replay validation even though they were correct when emitted. The dimension-policy question parallels severity-policy (#001 + the `severity_policies` table + `policy_version` field) but isn't yet resolved at the canonical layer.

The post-foundation review surfaced the question explicitly: keep the validator strict (forces immutability), version the mapping like severity (adds infrastructure), or drop the validator entirely (lets stored dimensions survive arbitrary mapping drift, weakens classification integrity). This entry resolves the canonical position.

**Decision.** `FINDING_TYPE_TO_DIMENSION` is **append-only for existing `FindingType` members in V1**. The model validator stays; the immutability is the canonical position, not an accident of the validator's strictness.

1. **New `FindingType` values may be added at any time.** The module-load lockstep guard in `outrider.policy.dimensions::verify_lockstep` requires a matching `FINDING_TYPE_TO_DIMENSION` entry in the same commit (and a matching `SEVERITY_POLICY` entry); the existing audit chain for new types is well-supported.

2. **Existing `FindingType → ReviewDimension` mappings cannot change.** If a mapping feels wrong later (e.g., a future judgment call that `MISSING_INPUT_VALIDATION` belongs to `BEST_PRACTICES` rather than `SECURITY`), the resolution is **add a new, more-precise `FindingType`** — never remap the existing one. Old audit rows continue to describe the world the way it was when classified; new rows use the new type.

3. **Changing a mapping requires a new `DECISIONS.md` entry AND a migration plan for historical audit rows.** Migration plans must address (a) whether to backfill `FindingEvent` rows under the new mapping (rewrite history) or leave them under the old one (preserve history + accept replay-validator failures for old rows under the new mapping), and (b) how to flag the boundary in the audit stream so dashboards / replay tools can render correctly. Neither path is supported in V1; this entry says "decide explicitly via a new entry if the need arises."

4. **`dimension` is product taxonomy, not policy.** Severity is a policy knob — it reflects risk appetite and changes as the organization's posture changes. Dimension is a stable classification of *what kind of concern this finding is* (security / performance / quality / test-coverage / best-practices). Product taxonomy is the kind of thing that should change rarely and only with explicit decision; building dimension-policy-versioning infrastructure (a `dimension_policies` table + `dimension_policy_version` field + versioned loader) parallel to severity would be too much machinery for a surface that, by design, isn't operationally mutable.

5. **The validator's "replay fails on drift" behavior is a feature, not a bug.** A replay that fails because `FINDING_TYPE_TO_DIMENSION` was changed without a #021-style amendment IS the lockstep guard catching a forbidden ontology rewrite. The error message points the operator at this decision entry so they know whether to revert the mapping change or land a new DECISIONS entry.

**Consequences.**

- `ReviewFinding._enforce_dimension_lockstep` stays as the runtime gate. Tests that exercise alternate dimensions construct ReviewFindings with `finding_type` values that legitimately map to the alternate dimension under the CURRENT mapping (never patch the module attribute).
- `FINDING_TYPE_TO_DIMENSION` in `src/outrider/policy/dimensions.py` is the canonical mapping. Any future change to an EXISTING key requires a new DECISIONS entry (call it `#021-amended` or a new number; either is fine — the workflow doesn't forbid amending #021 in place if the change is small).
- Adding a new `FindingType` lands in the same commit as the matching `SEVERITY_POLICY` + `FINDING_TYPE_TO_DIMENSION` entries. The lockstep guard fires at module load and at CI test time; no further infrastructure is needed.
- If a future requirement makes dimensions operationally mutable (V1.5+, a multi-tenant deployment that wants per-org dimension overrides, etc.), supersede this entry with a new one drafting the version-the-mapping infrastructure. Until then, immutability-by-decision is the canonical position.
- The audit-trail story stays clean: every `FindingEvent.dimension` at every point in audit history is the dimension the finding had at emission time AND matches the current mapping, because the mapping hasn't changed.

**Referenced from.** `src/outrider/policy/dimensions.py` (`FINDING_TYPE_TO_DIMENSION` + `verify_lockstep` + `lookup_dimension`), `src/outrider/schemas/review_finding.py::_enforce_dimension_lockstep` (the validator this entry justifies), `specs/2026-05-19-analyze-foundation.md` §6 (foundation spec section that originally introduced the mapping), foundation-wide data-integrity audit F2 (the audit finding that surfaced the immutability-vs-versioning question).

## 022. Proposal identity is PR/file-scoped, not raw-proposal-shape-global

**Status:** Accepted, 2026-05-20. Prose-clarified 2026-05-24 alongside #024 (the candidate field renamed from `candidate_path` to `import_string`; semantic logic below is unchanged but read references to `candidate_path` as the post-resolution file path produced at trace execution time — V1 via probe-resolve (`_candidate_paths_for` + `fetch_file_content_at`), V1.5+ via `coordinates.resolve_candidate_paths(import_string, import_root)` — rather than as a direct schema field).

**Post-#024 reading (added 2026-05-24 to reconcile body prose with the renamed field shape).** Per `DECISIONS.md#024` (Accepted 2026-05-24), `TraceCandidate.candidate_path` was renamed to `TraceCandidate.import_string` (dotted Python import string; V1 ships import-string-only). The decision logic below — `TraceCandidate` is provenance not a fetch directive; fetch dedup belongs in trace execution; recipe folds `source_file_path` — is unchanged. Prose references to `candidate_path` in the Decision text and Consequences below now describe **the resolved file path produced at trace execution time** (V1: probe-resolved via `_candidate_paths_for` + `fetch_file_content_at` per `#024` point 4; future filesystem-aware shape: `coordinates.resolve_candidate_paths(import_string, import_root)`), NOT the schema field (the schema field is now `import_string`). The "trace node groups by `candidate_path` and fetches each target once per round" language reads as "trace node groups by the resolver-output file path (V1: the probe-resolved path; V1.5+: the filesystem-resolved path through the ast_facts import registry) and fetches each target once." `compute_candidate_id`'s recipe input was correspondingly renamed from `candidate_path` to `import_string` per #024 (canonical-encoding key changed; canonical-encoding shape — sort_keys=True via canonicalize_for_hash — preserved; SHA-256 cryptographic properties unchanged). When in doubt about field names today, read #024; when in doubt about identity-scope semantics, read this entry.

**Context.** The original `compute_proposal_hash` recipe in `policy/canonical.py` (added by the foundation §1 schemas commit) folded 8 keys from `AnalyzeFindingProposalRaw` — finding_type, evidence_tier, query_match_id, trace_path, title, description, evidence, span (byte_start/byte_end) — but **omitted** the source file the proposal came from. Codex round-6 audit (medium confidence) surfaced the consequence: two analyze passes over different source files that emit logically-identical proposals (same finding_type, same span coordinates, same description text) produce **identical** `proposal_hash`. `TraceCandidate.source_proposal_hash` inherits the hash. `candidate_id = compute_candidate_id(source_proposal_hash, candidate_path, reason)` then collapses two distinct causal edges (File A → target T, File B → target T) into one row under the `append_with_dedup_by(candidate_id)` reducer.

Two readings of "what is a proposal":

- **Reading A — raw-proposal-shape-global (original behavior):** A proposal is "this shape of finding"; trace candidates dedup across source files so the trace node fetches the target file once even when multiple source files request it. Audit-trail loses the per-source-file causal edge as a known trade-off.
- **Reading B — PR/file-scoped (this decision):** A proposal is "this finding in this file." Trace candidates preserve per-source-file provenance; the trace node can still dedup actual fetches by `candidate_path` at execution time, but the candidate-identity model carries the causal edges intact.

**Decision.** Proposal identity is **PR/file-scoped, not raw-proposal-shape-global**. `compute_proposal_hash` gains a `source_file_path` parameter; the recipe folds 9 keys instead of 8. `TraceCandidate.source_proposal_hash` inherits the file-scoped hash; `candidate_id`'s derivation stays unchanged (`source_proposal_hash + candidate_path + reason`) because `source_proposal_hash` now carries source-file identity intrinsically.

The reasoning:

1. **`TraceCandidate` is provenance, not a fetch directive.** The audit trail's "which finding caused this trace request" question must remain answerable. If File A and File B independently raise findings that both point to `src/auth/middleware.py`, both causal edges belong in the audit log. Dedup that collapses them optimizes one read (avoiding double-fetch of the target) at the cost of losing the audit-grade record of two findings — that's the wrong trade-off for a system whose primary value-prop is audit-trail integrity.

2. **Fetch dedup belongs in trace execution, not in candidate identity.** The trace node can, after preserving candidate provenance, group candidates by `candidate_path` and fetch each target file once per analyze ⇄ trace iteration. That's a separate concern from how the audit layer records why the fetch happened. Conflating the two was the original mistake.

3. **Reading A was internally consistent but optimized away audit provenance.** The default for this project is "preserve audit provenance whenever the choice arises." That's a higher-order constraint than the local-optimization argument for global dedup.

4. **The recipe change is small and additive.** Adding `source_file_path` to `compute_proposal_hash` is a one-key recipe extension; the existing 8 keys are unchanged. Existing audit rows (none in production yet — this is the foundation arc, pre-V1-launch) are not affected because the foundation arc is still pre-production. Post-launch, the same shape would require a `proposal_hash_version` field or a migration; we land the right shape now while no production data exists.

**Consequences.**

- `compute_proposal_hash` signature gains `source_file_path: str` (keyword-only, leading the spec ordering since it's the new identity-scope key). The recipe folds it as the first key in the canonical-encoding dict so future readers see "file is part of identity" immediately.
- `FindingProposalRejectedEvent.proposal_hash` carries the new file-scoped digest. The event already stores `file_path` separately, so the AUDIT JOIN against the hash is unambiguous; this change tightens the dedup-key contract that consumers may rely on.
- `TraceCandidate.source_proposal_hash` carries the new file-scoped digest. The dedup-by-candidate_id reducer now collapses only candidates that came from the **same source file AND same proposal shape AND same target AND same reason** — the four-tuple that genuinely represents "the same causal edge."
- Trace execution gains an explicit responsibility: after the dedup-by-candidate_id reducer admits the candidates list, the trace node groups by `candidate_path` and fetches each target once per round. That logic lives in the trace-node spec (not the foundation arc), but the candidate-identity model is now shaped to enable it without conflating concerns.
- The fetch-optimization story for trace doesn't degrade: two findings from different source files pointing at the same target produce two `TraceCandidate` rows with distinct `candidate_id`s, but the trace node sees `candidate_path == src/auth/middleware.py` for both and fetches once. The audit log retains both causal edges; the GitHub API call count is unchanged.
- The original (Reading A) behavior is superseded; no historical audit rows exist that depend on it. Future contributors reading the recipe see the file-scoped shape in code AND the rationale here.

**Referenced from.** `src/outrider/policy/canonical.py::compute_proposal_hash` (the recipe this entry shapes), `src/outrider/schemas/trace_candidate.py` (`source_proposal_hash` field — inherits the file-scoped digest), `src/outrider/audit/events.py::FindingProposalRejectedEvent.proposal_hash` (audit-row carrier of the file-scoped digest), Codex round-6 audit MEDIUM #4 (the audit finding that surfaced the identity-scope question), the audit-the-audit user response confirming Reading B as the canonical position.

## 023. Publish routing and eligibility are separate decisions, not one combined gate

**Status:** Accepted, 2026-05-22.

**Amended 2026-05-27:** the "PublishEvent survives as the canonical review-level summary (one row per logical publication, including external-record recovery paths that don't re-emit)" clause in Consequences narrows. The `external-record recovery paths that don't re-emit` framing implicitly assumed the original successful process emitted PublishEvent before crashing — in which case the recovery path discovers the prior PublishEvent at Step 4 (intra-Outrider `query_prior_publish_event` returns it), the outcome is `IDEMPOTENTLY_SKIPPED`, and the prior PublishEvent IS the canonical record. The narrower path this clause silently elided is the **crash-after-POST-before-PublishEvent** scenario: the original process posted to GitHub successfully but died before persisting PublishEvent. The audit stream for that review has no PublishEvent at all, and recovery flows through Step 6's body-marker scan with outcome `IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD`. For this specific path, **`PublishAttemptEvent.recovered_github_review_id`** (new field added 2026-05-27 alongside this amendment) is the canonical github_review_id binding — replay tooling reads it from the attempt event because no paired PublishEvent ever lands. The amendment does NOT change architectural intent: PublishEvent still IS the canonical review-summary record for every path that emits one; the external-record-skip path's omission is now documented as intentional rather than an unfilled obligation. No new PublishEvent row is required on the recovery path — the recovered_github_review_id field + the attempt event's outcome discriminator together carry the full audit-replay contract.

**Context.** The §V publish-node design originally framed routing (where a finding goes — `INLINE_COMMENT` / `REVIEW_BODY` / `DASHBOARD_ONLY`) and materialization (whether the finding actually publishes — gated by severity + HITL availability) as one combined decision. A single `PublishRoutingEvent` carried both outcomes implicitly: a finding with no row was either "not yet processed" or "withheld at the gate" with no audit-grade way to tell. The publish-node feature spec (`specs/2026-05-21-publish-node.md`) Q3 surfaced the problem: when `hitl` is not yet shipped, `CRITICAL` / `HIGH` findings need a recorded withholding decision so the audit trail answers "why didn't this finding post" without requiring the reader to reconstruct policy-version-at-the-time from elsewhere. Combining the two decisions also conflates the trust boundary: routing is coordinate-system-semantic (lives in `coordinates/`), eligibility is policy-semantic (lives in `policy/`); collapsing them admits drift where a publisher-side override of "this critical finding actually got posted" looks identical to "this finding went inline because coordinates said so."

**Decision.** V1 publish emits **three event types per finding-or-attempt**, each carrying one orthogonal decision:

1. `PublishRoutingEvent` — coordinates-derived. For every finding the publish node processes, regardless of eligibility. Carries the `PublishRoutingReason` StrEnum (4 values: `reviewable_diff_line`, `unchanged_region`, `non_diffed_file`, `coordinate_error` — the last is an umbrella for any non-`UNCHANGED_REGION` / non-`FILE_NOT_IN_PATCH` `CoordinateError.kind`, *including* `PATH_VALIDATION_FAILED`; the kind itself rides on `coordinate_error_kind` so the audit stream can group by structural failure class without expanding the reason taxonomy), `coordinate_error_kind` (enum value ONLY, never `CoordinateError.message` text — info-leak defense for the `PATH_VALIDATION_FAILED` umbrella), identity tuple (`file_path`, `line_start`, `line_end`, `finding_type`, `finding_content_hash`), and `decision_content_hash` for consumer-side dedup-with-divergence-detection. Routing fires **even when** the finding is later withheld by the eligibility gate — the audit trail records that coordinates DID classify the finding, and what the classification was.

2. `PublishEligibilityEvent` — policy-derived. Fires together with `PublishRoutingEvent` under an interleaved per-finding loop (single pass, not two sequential — prevents a routing-loop crash from orphaning routing events without their eligibility partners). Carries `PublishEligibility` (`eligible` / `withheld`), `PublishEligibilityReason | None` (required iff withheld; V1 values: `hitl_required_node_absent`, `unexpected_override_fields_present`, `routing_emission_failed`), `severity`, `original_severity` (semantics narrowed by the 2026-05-27 amendment above — non-None is admissible when the gated finding carries a reviewer-issued `PerFindingDecision(outcome=SEVERITY_OVERRIDE)` via HITL; the producer-bug / replay-injected-fabricated-downgrade framing applied only PRE-HITL when no override path existed), `policy_version` (versioned for replay; live-policy match check fires ONLY when `policy_version == ACTIVE_POLICY_VERSION` per `severity-policy-versioned-for-replay`), and a `decision_content_hash` mirror. **Fabricated-override defense is two-layered with distinct responsibilities** (per the schema-vs-gate authorization-split clarified 2026-05-27): (a) the schema validator `_enforce_override_legitimacy` in `audit/events.py` enforces only the **row-local invariant** that the audit row's `severity` and `original_severity` are consistent with the override claim (a real `SEVERITY_OVERRIDE` implies `severity != original_severity`; a no-op override where the two match is rejected at construction) — the schema validator CANNOT consult HITLDecision context because that's a separate event the validator doesn't see; (b) the gate function `is_eligible_for_v1_publish` in `policy/publish_eligibility.py` performs the **cross-event HITL authorization check** before materialization — it receives `hitl_request` + `hitl_decision` as explicit kwargs and verifies that any non-None `original_severity` on the finding is backed by a matching `PerFindingDecision(outcome=SEVERITY_OVERRIDE)` keyed on `finding_id` AND that the finding was in the gated set of the request (returning `withheld + reason=unexpected_override_fields_present` otherwise — see `FOLLOWUPS.md` FUP-062). Both layers are required for the trust story: the schema guard catches row-local corruption (severity == original_severity with override claim); the gate catches cross-event forgery (override claim without matching HITLDecision). **`severity-set-by-policy` invariance:** the gate is where authorization lives because it's the deterministic system that consults all the relevant context; putting cross-event authorization in the audit schema would conflate schema-validation (row-local correctness) with policy-enforcement (cross-event severity-policy adherence).

3. `PublishAttemptEvent` — GitHub-call outcome. Single emission per `publisher.create_review` attempt, AFTER the call resolves (no in_flight pre-call emission — would conflict with append-only audit semantics under same-`event_id`-different-payload). Carries `PublishAttemptOutcome` (`success` / `failed` / `idempotently_skipped` / `idempotently_skipped_external_record` / `no_op_empty`), `failure_class` (required iff `failed`, must be None otherwise), and an `attempt_content_hash` that includes `outcome` so success-then-failed-replay rows distinguish in consumer dedup. Two distinct idempotent outcomes split the crash-recovery story: `idempotently_skipped` fires when a prior `PublishEvent` for the `review_id` is found in the local audit log (the pre-flight check short-circuits before any GitHub call); `idempotently_skipped_external_record` fires when no prior `PublishEvent` exists but `find_existing_review_on_head_sha` finds a matching review on GitHub via the `<!-- outrider-review-id:{review_id} -->` body marker (the crash-after-success path — the prior process succeeded at the GitHub call but died before persisting `PublishEvent`). `no_op_empty` fires when zero eligible+INLINE findings remain after the routing/eligibility loop; no GitHub call made.

**V1 implementation status (amended 2026-05-22 post-FUP-064 closure).** All FIVE `PublishAttemptOutcome` values are live producer outcomes in V1 publish-node code: `success`, `failed`, `idempotently_skipped` (FUP-064 closed — `AuditPersister.query_prior_publish_event(review_id)` ships as the first read-side method on the class; publish node calls it at Step 4 BEFORE the empty-eligible short-circuit and BEFORE the external-record check, per spec §V pre-flight ordering), `idempotently_skipped_external_record` (crash-after-success path via body-marker query on GitHub), `no_op_empty`. All five enum `.value` strings remain golden-pinned by `tests/unit/test_publish_decision_hash_recipes_pinned.py`. The intra-Outrider check at Step 4 is the canonical defense against same-`review_id` redispatch; the external-record check at Step 6 covers the narrower crash-after-success window where the prior process succeeded at the GitHub POST but died before persisting `PublishEvent`. Unit tests pinning the producer paths live at `tests/unit/test_publish_idempotency.py` (FUP-066 closed).

**Consequences.**

- **Three orthogonal audit signals.** "Did coordinates classify this finding?" "Did the policy gate admit it?" "Did the GitHub call succeed?" are now three separate yes/no questions answerable from three rows, not one combined row that requires inference. Replay equivalence per `audit-replay-equivalence-window` reconstructs each independently.
- **HITL-gate absence is recorded, not implicit.** Until `hitl` ships, every `CRITICAL` / `HIGH` finding produces a `PublishEligibilityEvent(eligibility=withheld, reason=hitl_required_node_absent)` row paired with a normal `PublishRoutingEvent` row. The audit trail proves the finding was processed, classified, and withheld — not lost in the pipeline. When `hitl` ships, the gate flips to consult `HITLDecisionEvent` for these findings; the eligibility-event shape is unchanged.
- **Routing-emission-failure recovery is auditable.** A per-finding `try/except` around `emit_publish_routing` falls back to `PublishEligibilityEvent(eligibility=withheld, reason=routing_emission_failed)`. Without the decoupling, a failed routing emission would lose the finding silently; with it, the failure is explicit.
- **Trust-boundary fidelity.** Routing decisions live in `coordinates/`; eligibility decisions live in `policy/`. The publisher orchestrates and emits, but does not produce either decision. A future change that wants to "let the publisher override an eligibility gate" hits the schema before it hits the bug — `original_severity is None` is enforced at construction.
- **Cost.** Three audit rows per finding (routing + eligibility + at most one attempt at the review level) is ~3× the row count of a combined-event design. Realistic per-PR audit volume (median 5-15 findings) puts this in the tens of rows, dominated by other event types. Acceptable.
- **`PublishEvent` survives** as the canonical review-level summary (one row per logical publication, with one exception — the **crash-after-POST-before-PublishEvent recovery path** described in the 2026-05-27 amendment above, where no PublishEvent lands and `PublishAttemptEvent.recovered_github_review_id` is the canonical github_review_id binding). It does not duplicate the per-attempt detail in `PublishAttemptEvent`; it records `(github_review_id, comments_posted, review_status)` for the outcome the dashboard surfaces. The two event types compose: an operator reconstructing a failed run reads `PublishAttemptEvent(outcome=failed)` + absence of `PublishEvent` as the dangling-failure signal; the operator distinguishes the dangling-failure case from the IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD recovery case by the `PublishAttemptEvent.outcome` discriminator + the presence of `recovered_github_review_id`.
- **Append-only enum-value + hash-recipe contract.** The four publish StrEnums (`PublishRoutingReason`, `PublishEligibility`, `PublishEligibilityReason`, `PublishAttemptOutcome`) and the three canonical decision-hash helpers (`compute_publish_routing_decision_hash`, `compute_publish_eligibility_decision_hash`, `compute_publish_attempt_content_hash`) are **append-only V1 contracts**: enum `.value` strings, member ordering relevance, and hash recipes (JSON encoding order, separators, included fields) are pinned by this entry. New enum members may be added; existing members and their `.value` strings MUST NOT be renamed, removed, or reordered. Hash recipes MUST NOT change input shape. Rationale: every hash validator (`_verify_finding_content_hash`, `_verify_decision_content_hash`, `_verify_content_hash_binding`, `_verify_attempt_content_hash`) recomputes at construction and rejects on mismatch; renaming `reviewable_diff_line` → `inline_eligible` or reordering the JSON-array recipe would silently break every historical row's `model_validate` at replay. The lighter alternative (a `hash_recipe_version` field with replay-aware skip-for-historical, mirroring `PublishEligibilityEvent.policy_version`'s `ACTIVE_POLICY_VERSION` guard) is deferred to V1.5 if the contract ever needs to evolve; today's discipline is "don't evolve the contract."
- **Anomaly-flooding cap** (`FOLLOWUPS.md` FUP-063): distinct `decision_content_hash` values per `(review_id, finding_id)` cap at 3-5 at write-time to prevent attacker-flooding-via-decision-drift; after cap, a single `decision_drift_saturated` event lands instead of the N-th drift row. The FUP has trigger (V1.5 dashboard work begins) and exit (persister-side cap + saturation-transition test) — see FUP-063 for the full contract.

**Referenced from.** `docs/spec.md` §V (publish routing summary), `docs/spec.md` §8.2 (event-type table for `PublishRoutingEvent`, `PublishEligibilityEvent`, `PublishAttemptEvent`, `PublishEvent`), `src/outrider/audit/events.py` (the four event classes, four StrEnums — `PublishRoutingReason`, `PublishEligibility`, `PublishEligibilityReason`, `PublishAttemptOutcome`, and three canonical decision-hash helpers — `compute_publish_routing_decision_hash`, `compute_publish_eligibility_decision_hash`, `compute_publish_attempt_content_hash`), `specs/2026-05-21-publish-node.md` (the feature spec that motivated this entry), `src/outrider/coordinates/errors.py::CoordinateErrorKind` (the structural taxonomy whose enum values land in `PublishRoutingEvent.coordinate_error_kind`).

## 024. Trace candidates are dotted Python import strings (V1)

**Status:** Accepted, 2026-05-24. Amends #017 (event/state field names + cross-field validator rules — full delta carried inside this entry and summarized on #017's amendment line).

**Context.** The trace-node spec (`specs/2026-05-23-trace-node.md`) first-round audit (AUDIT_LOG.md 2026-05-24) raised an unresolved question: `TraceCandidate.candidate_path` is a normalized repo-relative file path (validated by `validate_diff_path` at construction), but `docs/architecture.md` line 30 describes trace as resolving candidates "through the ast_facts import registry" — and `coordinates.resolve_candidate_paths(import_string: str, import_root: Path) -> list[Path]` (the `ImportPathResolver` Protocol implementation) consumes dotted Python import strings, not file paths. The two surfaces don't connect. A spec-fidelity audit pass pushed back on the XOR/fallback option that admits both forms: "LLM proposes a path and proves it is not import-resolvable" is not a deterministic proof boundary, and a fallback branch reopens the determinism gap the resolver was supposed to close. The decision settles trace's V1 candidate shape AND, as a load-bearing consequence, amends `DECISIONS.md#017`'s event/state field naming + validator rules so the once-per-finding contract continues to hold under import-string semantics.

**Decision.** Trace candidates are dotted Python import strings in V1. No file-path fallback.

1. **`TraceCandidate.import_string: str` replaces `candidate_path`.** The schema carries a single dotted-form field (e.g., `"foo.bar"`). The field validator runs the same identifier-validity + shell-metacharacter + separator-rejection checks `resolve_candidate_paths` runs at call time (per `coordinates/diff_parser.py:162-178`), lifted to schema construction so malformed forms are rejected at the producer's emission site, not the resolver's call site. Pattern: dot-separated parts; each part is a valid Python identifier (per `str.isidentifier()`) and not a Python keyword; no backslash or forward slash; no shell metacharacters. `compute_candidate_id` recipe input changes from `(source_proposal_hash, candidate_path, reason)` to `(source_proposal_hash, import_string, reason)` — recipe stays content-derived; the canonical-encoding shape uses `import_string` in place of `candidate_path`.

2. **Same-file candidates do NOT flow through `TraceCandidate`.** Analyze handles same-file references inline at admission time via the parsed scope-unit graph in `ast_facts/`. Per architecture line 30: "same-file candidates skip this." V1 enforces "skip this" as "never construct a `TraceCandidate` for them" — trace's input is exclusively cross-file references. The scope-unit graph is in-memory in analyze; cross-trace forensic value of same-file candidates is zero (analyze already has the resolution).

3. **No file-path fallback.** The XOR shape (`import_string | None` and `candidate_path | None` with cross-field "exactly one" validator) was rejected as an attractive nuisance. The fallback would absorb every case where the LLM emitted the wrong shape, defeating the resolver's role as deterministic boundary. If V1.5+ identifies a real need for path-literal trace (e.g., for data files or non-Python languages where import semantics don't apply), it ships as its own resolver with a tagged identity hash and a separate `TraceCandidate`-sibling schema, not a hidden second branch in V1.

4. **Trace's resolver call shape:** `resolved = resolver.resolve_candidate_paths(candidate.import_string, import_root)`. `len(resolved) == 0` → `resolution_status = "unresolved"`; `len(resolved) == 1` → `resolution_status = "resolved"`; `len(resolved) > 1` → `resolution_status = "ambiguous"`. Maps directly to `TraceDecision.resolution_status` (per #017 amended point 3).

   **Amended 2026-05-24:** the resolver call shape above is the FUTURE filesystem-aware shape (V1.5+ when local-checkout architecture lands). The V1 implementation per `specs/2026-05-23-trace-node.md` M8 uses a **two-phase GitHub-API fetch** instead: trace constructs candidate paths from the import string via `agent/nodes/trace.py::_candidate_paths_for` (`foo.bar` → `foo/bar.py` + `foo/bar/__init__.py`), validates each via `coordinates.validate_diff_path`, then issues Phase 1 fetch-probes via `github.fetch.fetch_file_content_at` at head SHA. The Phase 1 probe-outcome enumeration maps the same way (`len(resolved) == 0/1/>1` → `unresolved`/`resolved`/`ambiguous`); the resolver MECHANISM differs (filesystem stat vs GitHub fetch-probe) but the resolution-status semantics + downstream TraceDecisionEvent shape are unchanged. The amendment narrows the call-shape clause to its filesystem-aware future scope; the rest of #024 (import-string identity, field renames, validators, audit-shadow rules) stays intact. See M8 for the full V1 two-phase fetch design + cost-shape analysis + V1.5 evolution path.

5. **Amendment to #017 (carried inline, same working change as this entry).** The renaming-and-splitting required to make import-string candidates fit #017's cross-field validators. #017's core commitment (one decision per `source_finding_id`, reducer key on `source_finding_id` alone) is unchanged; the field shape underneath is.

   Field renames + additions on both `TraceDecision` (schema layer) AND `TraceDecisionEvent` (audit-event mirror per the audit-shadow rule in `docs/conventions.md` "Audit event emission"):

   - `candidates_considered: tuple[str, ...]` → `proposed_import_strings: tuple[str, ...]` (the LLM-proposed dotted forms; set-semantic; unique).
   - Add `resolved_candidate_paths: tuple[str, ...]` (resolver outputs from `resolve_candidate_paths`; set-semantic; unique; post-`validate_diff_path` form per the audit-shadow rule — see point 6).

   Validator rules (replaces the three rules in #017 amended point 3 clauses (a)/(b)/(c)):

   - `resolution_status == "resolved"` → `len(resolved_candidate_paths) == 1` AND `target_file == resolved_candidate_paths[0]`
   - `resolution_status == "unresolved"` → `len(resolved_candidate_paths) == 0` AND `target_file is None`
   - `resolution_status == "ambiguous"` → `len(resolved_candidate_paths) > 1` AND `target_file is None`

   The existing uniqueness validator (`_enforce_candidates_considered_unique`, `audit/events.py:693`) splits into two — one for `proposed_import_strings`, one for `resolved_candidate_paths`. Both tuples are set-semantic; duplicates inside either are a producer bug.

6. **Path-bearing field audit-shadow.** Both `target_file` (when non-None) AND every element of `resolved_candidate_paths` pass through `validate_diff_path` at the audit-event boundary (`TraceDecisionEvent` field validators). The resolver already produces safe repo-relative paths, but the audit-event schema must shadow the boundary the same way other path-bearing events do — defense in depth at the append-only log, against a hypothetical future direct emitter that bypasses the resolver. Same shape as `FindingEvent._enforce_canonical_file_path` (`audit/events.py:504`). The per-element rule on `resolved_candidate_paths` is load-bearing for the `ambiguous` branch specifically: `target_file is None` for ambiguous decisions, but the tuple still carries multiple resolver-output paths that land in the append-only log and would otherwise enter audit storage unvalidated. Same per-element shadow on the schema-layer `TraceDecision.resolved_candidate_paths`.

**Consequences.**

- `src/outrider/schemas/trace_candidate.py` — drop `candidate_path`; add `import_string: str` with the identifier/separator/metacharacter field validator. `compute_candidate_id` recipe input changes accordingly. Existing `candidate_id`-bearing tests update fixtures.
- `src/outrider/schemas/llm/analyze.py` — `TraceCandidateProposalRaw` + `TraceCandidateProposal` carry `import_string`, not `candidate_path`. Raw-layer admission rejects path-shaped proposals (route to a new `FindingProposalRejectedEvent.rejection_reason` value if attached to a finding, else drop the candidate).
- `src/outrider/audit/events.py::TraceDecisionEvent` (`audit/events.py:654`) — field rename (`candidates_considered` → `proposed_import_strings`) + new field (`resolved_candidate_paths`). Validator rewrite per point 5. New `validate_diff_path` shadow on `target_file` per point 6. Uniqueness validator split per point 5.
- Schema-layer `TraceDecision` (lands with the trace-node spec at `schemas/review_state.py:35`) — same shape changes, same validators.
- `src/outrider/agent/nodes/analyze.py` — admission gate stops stamping `candidate_path`; produces `import_string` for cross-file candidates; handles same-file candidates inline via scope-unit graph lookup (no `TraceCandidate` emission for same-file refs).
- `src/outrider/agent/nodes/trace.py` (when written) — resolver call shape per point 4; `TraceDecisionEvent` construction populates both `proposed_import_strings` and `resolved_candidate_paths` per the amended #017 shape.
- `src/outrider/prompts/templates/user/analyze.md` (or analogue) — instruction text changes from "candidate path" to "dotted Python import string for cross-file references; omit for same-file references; do NOT emit file paths."
- `tests/unit/test_trace_candidate.py` + sibling tests — XOR/admit tests pivot to `import_string`; new schema-validator tests for the identifier/separator rejection rules.
- `tests/unit/test_audit_events.py` (`TraceDecisionEvent`) — validator-rewrite tests; resolved/unresolved/ambiguous admit tests with the new field shape.
- `docs/spec.md` §4.1.4 (TraceDecision) + §7.1 (`TraceCandidate` in `ReviewState.trace_candidates`) + §8.2 (TraceDecisionEvent) — wording amended in lockstep: candidates are import strings; events carry parallel proposed/resolved tuples.
- `docs/architecture.md` line 30 — wording stays correct ("import registry" framing matches this decision).
- `specs/2026-05-19-analyze-foundation.md` Actual Outcome — addendum noting `TraceCandidate.candidate_path` shipped under a since-superseded design and was renamed under #024.

**Migration.** None. `TraceCandidate` has no production consumer (analyze emits the field but no trace consumer exists); the schema rename ships atomically with trace's first commit. `audit_events` table contains no `TraceDecisionEvent` rows (trace hasn't shipped); the `TraceDecisionEvent` field rename + add is a Pydantic schema change with no backfill needed (per #018 point 5: `audit_events.payload` is JSONB; no DB-level constraint).

**No supersession of prior DECISIONS.** #017 is amended, not superseded — its core commitment (one decision per source finding, reducer key on `source_finding_id`) is unchanged. The amendment marker on #017 preserves the chain.

**Referenced from.** `src/outrider/schemas/trace_candidate.py` (`TraceCandidate.import_string` + field validator + `compute_candidate_id` recipe), `src/outrider/schemas/llm/analyze.py` (`TraceCandidateProposalRaw`, `TraceCandidateProposal`), `src/outrider/audit/events.py` (`TraceDecisionEvent` field rename + add + validator rewrite + `target_file` audit-shadow), `src/outrider/agent/nodes/analyze.py` (admission path; same-file inline handling), `src/outrider/agent/nodes/trace.py` (when written — resolver call shape; event construction), `src/outrider/coordinates/diff_parser.py` (`resolve_candidate_paths` — the consumed resolver surface; pattern-validation rules lifted to the schema), `docs/spec.md` §4.1.4 + §7.1 + §8.2, `docs/architecture.md` line 30, `specs/2026-05-23-trace-node.md` (consumer of this decision), `specs/2026-05-19-analyze-foundation.md` Actual Outcome (addendum on the `candidate_path` rename).

## 025. Admitted findings carry `proposal_hash`; trace gates on join + once-per-finding

**Status:** Accepted, 2026-05-24.

**Context.** Trace's audit-event contract per `DECISIONS.md#017` requires `TraceDecisionEvent.source_finding_id: UUID` — a stable identifier from the admitted-finding branch. But `TraceCandidate.source_proposal_hash` (per `schemas/trace_candidate.py:53`) is a proposal-layer identifier, and the schema docstring explicitly admits candidates from REJECTED proposals ("a rejected JUDGED-claim might still surface a legitimate cross-file-to-look-at signal"). Verified during the trace-spec audit (AUDIT_LOG.md 2026-05-24): `ReviewFinding` carries no `proposal_hash`; `FindingEvent` carries no `proposal_hash`; the only `proposal_hash`-bearing event is `FindingProposalRejectedEvent` (the rejected branch only). There is no existing field that joins `TraceCandidate.source_proposal_hash` to an admitted `finding_id`. Three options were enumerated in the trace spec; the spec-fidelity audit settled on extending admitted findings to carry `proposal_hash` as the only option that preserves the rejected-proposal candidate-collection rule, restores the join, and does NOT require superseding #017's carefully-amended once-per-finding shape.

**Decision.** Admitted findings carry `proposal_hash` so trace can join `trace_candidates` to admitted `finding_id`s. Trace's emission gate consults BOTH the join AND once-per-finding.

1. **`ReviewFinding.proposal_hash: str`.** New required field, pattern-validated against `SHA256_HEX_PATTERN`. Carries the same digest analyze's admission gate computes via `compute_proposal_hash` (per `DECISIONS.md#022`). No default — analyze's admission path constructs the field at the same call site `FindingProposalRejectedEvent.proposal_hash` is stamped today; the link is just kept on the admitted branch too. No new model_validator (the pattern + no-default contract are the gate).

2. **`FindingEvent.proposal_hash: str`.** Audit-shadow mirror per `docs/conventions.md` "Audit event emission" — the audit event carries at least the source schema's fields so replay reconstruction can join trace decisions back to finding events without consulting `ReviewFinding`. Same pattern validation; same no-default contract.

3. **`proposal_hash` is provenance, NOT content identity.** It does NOT enter `compute_finding_content_hash` — the recipe stays `(file_path, line_start, line_end, finding_type)` per the canonical recipe in `audit/events.py`. Two analyze passes that produce the same logical finding from differently-shaped proposals still collide on `finding_content_hash` (correct — same content) but carry distinct `proposal_hash` values (correct — different provenance). The provenance/identity split is the same shape `DECISIONS.md#022` established for proposals: identity over provenance pollutes content-derived dedup; provenance over identity loses the audit chain. Keep them separate.

4. **Schema contract: `proposal_hash` is unique across admitted findings within a review.** Enforced by:
   - `AnalysisRound._enforce_findings_proposal_hash_unique` model_validator (mode="after"): within-round invariant, runs cheaply on every reducer merge — `assert len({f.proposal_hash for f in findings}) == len(findings)`.
   - Cross-round uniqueness enforced at trace's join-construction site (see point 5) plus an analyze-side admission test (`tests/unit/test_analyze_node.py`) that pins the cross-round invariant.

5. **Trace's emission gate.** Trace emits `TraceDecisionEvent` AND triggers `github.fetch` for a candidate if and only if BOTH of:
   - **Join gate.** `candidate.source_proposal_hash` is in the lookup `{f.proposal_hash → f.finding_id}` built from `state.analysis_rounds`. Lookup construction raises `TraceJoinIntegrityError` on collision (loud fail; never silently last-wins — a collision indicates the uniqueness invariant from point 4 broke upstream, and trace masking it would hide the bug).
   - **Once-per-finding gate.** The joined `finding_id` is NOT in `already_traced: set[UUID] = {d.source_finding_id for d in state.trace_decisions}`. This makes #017's "one decision per source finding across the review" explicit at the emission path, not just emergent from the reducer key.

6. **Unjoined candidates remain forensic-only.** A `TraceCandidate` whose `source_proposal_hash` is not in the lookup (rejected parent proposal; replay-order edge cases in a hypothetical V1.5 parallel-analyze) stays in `state.trace_candidates` for replay but produces no `TraceDecisionEvent` and no GitHub fetch. The dedup-by-`candidate_id` reducer keeps the entry idempotent across replay.

**Why both gates, not just one (point 5).** The reducer dedup-on-`source_finding_id` makes `state.trace_decisions` idempotent under checkpoint replay — but `TraceDecisionEvent` is append-only to `audit_events` with a fresh `event_id` per emission per the canonical `AuditEventBase.event_id = default_factory=uuid4` policy. A re-invocation of trace post-checkpoint without the `already_traced` gate would emit fresh duplicate `TraceDecisionEvent` rows for findings already traced (reducer would collapse the `TraceDecision`, audit-persister would NOT collapse the event since `event_id` differs). The gates reinforce each other: reducer enforces state idempotency; trace's emission gate enforces audit idempotency. Neither substitutes for the other.

**Consequences.**

- `src/outrider/schemas/review_finding.py` — add `proposal_hash: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]`. No new model_validator (pattern is the gate).
- `src/outrider/audit/events.py::FindingEvent` (`audit/events.py:469`) — add the same field, same pattern validator. Audit-shadow rule.
- `src/outrider/schemas/analysis_round.py` — add `_enforce_findings_proposal_hash_unique` model_validator (mode="after").
- `src/outrider/agent/nodes/analyze.py` — admission path threads `proposal_hash` from the raw-layer `compute_proposal_hash(...)` output through `ReviewFinding` construction. The hash is already computed at admission for `FindingProposalRejectedEvent`; this keeps it on the admitted branch too.
- `src/outrider/agent/nodes/trace.py` (when written) — explicit collision-detecting lookup builder + `already_traced` gate; raises `TraceJoinIntegrityError` on duplicate `proposal_hash` at lookup construction.
- `tests/unit/test_review_finding.py` + `tests/unit/test_audit_events.py` — new field admit + pattern-rejection tests.
- `tests/unit/test_analyze_node.py` — pin cross-round `proposal_hash` uniqueness for admitted findings.
- `tests/unit/test_analysis_round.py` (new or existing) — pin the within-round validator from point 4.
- `tests/unit/test_trace_node.py` (lands with trace impl) — pin loud-fail on duplicate `proposal_hash` lookup; pin once-per-finding gate behavior under simulated replay.
- `docs/spec.md` §7.3 (ReviewFinding) + §8.5 (FindingEvent) — wording amended to list `proposal_hash` as a required field; explicit note that it is provenance, not part of `finding_content_hash` (point 3).

**Migration.** None. The `audit_events` table contains no historical `FindingEvent` rows (analyze hasn't shipped to production per the same rationale `DECISIONS.md#018` point 5 cites). Pydantic field addition with no default is a hard contract: all existing test fixtures need updating in the same commit. That is an acceptable atomic cost; the alternative (`default=""`) admits empty-string proposal_hashes which would break the join silently at trace-emission time.

**No supersession of prior DECISIONS.** #017's commitment to one decision per source finding is preserved by point 5's `already_traced` gate (explicit, not merely emergent from the reducer key). #022's PR/file-scoped proposal-identity rule is preserved by point 3's separation of `proposal_hash` (provenance) from `finding_content_hash` (content identity).

**Referenced from.** `src/outrider/schemas/review_finding.py` (`ReviewFinding.proposal_hash`), `src/outrider/schemas/analysis_round.py` (`_enforce_findings_proposal_hash_unique`), `src/outrider/audit/events.py` (`FindingEvent.proposal_hash`), `src/outrider/agent/nodes/analyze.py` (admission threading), `src/outrider/agent/nodes/trace.py` (when written — join gate + `already_traced` gate + `TraceJoinIntegrityError`), `docs/spec.md` §7.3 + §8.5, `specs/2026-05-23-trace-node.md` (consumer of this decision).

## 026. Audit-event idempotency mode: `event_id`-PK vs natural-key

**Status:** Accepted, 2026-05-24.

**Context.** The `audit_events` table has, until trace-node, used a single idempotency mechanism: each event carries a UUID `event_id` generated per-emission via `AuditEventBase.event_id = default_factory=uuid4`; the persister's `_persist_non_phase_event` writes with `ON CONFLICT (event_id) DO NOTHING` and raises `AuditPersisterIdempotencyConflict` on payload divergence under PK collision. This works for events that are naturally unique per emission: `FindingEvent` (one per admitted finding), `LLMCallEvent` (one per provider call), `PublishRoutingEvent` (one per routing decision per finding). Each retry/replay produces a fresh `event_id`; PK conflict only fires on the same logical write repeating with the EXACT same UUID (rare; mostly LangGraph checkpoint replay edge cases).

Trace introduces a different shape. Per `specs/2026-05-23-trace-node.md` M7 + `DECISIONS.md#017`: a `TraceDecisionEvent` represents the trace node's logically-once decision for a given `source_finding_id` within a review. The audit-first emission contract requires that retry/replay of trace (transient sink failure, partial-loop failure, checkpoint resume) MUST produce no duplicate audit row even though each emission attempt mints a fresh `event_id`. The event_id-PK mechanism cannot enforce this — the natural identity is `(review_id, source_finding_id)`, not `event_id`. A second concern surfaced during the trace spec arc's audit rounds: when the persister no-ops on natural-key match, the producer node MUST be able to construct the state-layer mirror from the PERSISTED row's payload (not from the freshly-computed inputs), otherwise per-emission fields excluded from the identity subset (LLM-narrative text, ranking order, timestamps) cause state-vs-audit divergence on retry.

**Decision.** Audit events choose ONE of two idempotency modes at design time, and the mode is pinned in the event-class docstring + persister helper.

1. **`event_id`-PK idempotency** (existing mechanism, default for most events). Use when:
   - The event represents a discrete observable operation that is naturally unique per emission (an LLM call, a routing decision, a phase boundary, an admitted finding).
   - Retry/replay producing a fresh `event_id` is acceptable behavior — the consumer-side dedup (via `decision_content_hash` per #023 or `finding_content_hash` per the audit-events module spec) handles read-time deduplication.
   - The persister uses `_persist_non_phase_event` (or its event-type-keyed variants).

2. **Natural-key idempotency** (new mechanism, introduced by trace M7). Use when **duplicate semantic rows would violate an audit-first or state-completeness contract** — not merely "you can name a structural tuple" (publish events qualify under that loose framing but correctly stay event_id-PK; the consumer-side `decision_content_hash` dedup is sufficient for publish's read-time queries). The trigger condition for natural-key is specifically: a producer's write-time contract requires "audit row exists before state delta returns" AND retries/replay must not create duplicate audit rows AND state must stay in lockstep with the persisted audit row across retries.

   The event type ships FOUR coupled components:

   - **(a) Alembic migration** adding a partial unique index on the natural-key tuple filtered by `event_type = '<discriminator>'`.
   - **(b) Persister helper** `_persist_keyed_by_natural_key` (or equivalent) using `postgresql_insert(...).on_conflict_do_nothing(index_elements=[...], index_where=...)` — **NOT** raw INSERT + `IntegrityError(UniqueViolation)` catch. The `on_conflict_do_nothing` path is the existing persister idiom (`persister.py:1010`) and avoids the savepoint/transaction-rollback footguns of exception-driven conflict handling. On the no-rows-returned path (conflict fired), run a follow-up SELECT on the natural-key tuple to load the existing row, then compare against incoming via the identity-subset (component d).
   - **(c) Persisted-payload return contract.** The helper RETURNS the canonical persisted event payload — either the just-inserted event (insert path) OR the existing row's event (no-op path on identity-subset match). The sink Protocol method's signature is `async def emit_X(self, event: XEvent) -> XEvent: ...` (non-None return). The producer node MUST use the returned event to construct any state-layer mirror; this is the lockstep-recovery contract that keeps state in sync with audit when per-emission fields (LLM-narrative, ranking order, timestamps) differ between attempts. Without this, the crash-after-audit-before-state scenario diverges state from audit on retry. (Trace's M7 spells the `trace_decision` instance: state-layer `TraceDecision` is built from the returned `TraceDecisionEvent`, not from trace's locally-computed inputs.)
   - **(d) `_payload_identity_subset(event_type) -> frozenset[str]`** enumeration. Each natural-key event type lists its identity fields explicitly; per-emission fields are explicitly excluded.

   The persister raises a distinct **`AuditPersisterNaturalKeyConflict`** (NOT `AuditPersisterIdempotencyConflict`) on real divergence — the natural-key conflict carries both the existing row's PK and the conflicting natural key.

3. **First normative identity subset (pins the precedent for future natural-key event types):** for `trace_decision`, the identity subset is exactly:

   ```python
   _PAYLOAD_IDENTITY_SUBSET = {
       "trace_decision": frozenset({
           "source_finding_id",   # natural-key payload component
           "target_file",         # deterministic resolution outcome
           "resolution_status",   # outcome class (resolved/unresolved/ambiguous)
           "is_eval",             # invariant per review; divergence = config bug
       }),
   }
   ```

   Explicitly EXCLUDED (each would defeat the lockstep contract on legitimate retries):

   - `event_id` — per-emission UUID.
   - `timestamp` — per-emission datetime (NOTE: field is `timestamp`, NOT `emitted_at`).
   - `sequence_number` — per-emission (already excluded by `_serialize_event_payload`).
   - `review_id` — pinned by the natural-key index's lookup columns (the SELECT for the no-op recovery filters on `(review_id, payload->>'source_finding_id')`); tautological in the value-comparison.
   - `reason` — LLM-narrative; retries produce fresh Haiku-generated reasons.
   - `proposed_import_strings` — LLM ranking order varies across retries.
   - `resolved_candidate_paths` — derived from `proposed_import_strings`.
   - `trace_path` — per-emission scope-walk context.
   - `event_type` — pinned by the partial index's WHERE clause filter; tautological.

   Future natural-key event types extend the enumeration in a single commit with the event class. Subset choices are golden-pinned by `tests/unit/test_audit_persister_identity_subsets.py`.

4. **The choice is per-event-type, pinned in the event class's module docstring** — `Idempotency mode: event_id-PK` or `Idempotency mode: natural-key (key=(...))`. Mixing modes across the table is supported by design — the partial unique index from mode (2) is event-type-filtered, so it doesn't constrain mode (1) events.

5. **Selection rule.** Default to **event_id-PK** unless an audit-first / state-completeness contract requires natural-key. Specifically:
   - Choose **event_id-PK** when: the event records a discrete observable operation, the consumer-side dedup (via content-hash carried IN the payload per #023's `decision_content_hash` pattern) is sufficient for replay-equivalence queries, and retry/replay producing additional rows is acceptable.
   - Choose **natural-key** when ALL of: (i) a producer node's write-time contract requires "audit row exists in `audit_events` before state delta merges"; (ii) retry/replay must NOT create duplicate rows at the persister layer; (iii) state must stay in lockstep with the persisted audit row across retries (per-emission divergence on `reason`/timestamps/rankings would otherwise produce drift). Trace's M7 is the V1 instance — meets all three.

**Consequences.**

- `audit/events.py` event class docstrings gain an `Idempotency mode:` annotation naming `event_id-PK` or `natural-key (key=...)`.
- New persister helpers (`_persist_keyed_by_natural_key`) and exception classes (`AuditPersisterNaturalKeyConflict`, `AuditPersisterTraceIdempotencyLookupError`) land with trace per `specs/2026-05-23-trace-node.md` M7. Both helpers coexist; per-event-type routing in `AuditPersister.emit_*` methods selects.
- Sink Protocols for natural-key event types ship with non-None return signatures per point (2c). `TraceEventSink.emit_trace_decision(event) -> TraceDecisionEvent` is the V1 first instance.
- The `_payload_identity_subset` enumeration is golden-pinned per event type by `tests/unit/test_audit_persister_identity_subsets.py`. Integration tests additionally pin the persisted-payload-return contract (no-op path returns existing row's event; insert path returns incoming event).
- Future migrations can mix modes per event type — no global re-architecture; the table's PK + per-event-type partial indexes co-exist.

**Referenced from.** `src/outrider/audit/events.py` (event class `Idempotency mode:` docstring annotations — added per-type as event types land), `src/outrider/audit/persister.py` (`_persist_non_phase_event` + new `_persist_keyed_by_natural_key`), `src/outrider/audit/sinks.py` (sink Protocols with non-None return for natural-key event types), `src/outrider/db/models/audit_events.py` (`event_id` PK + partial unique indexes from per-event-type migrations), `specs/2026-05-23-trace-node.md` M7 (first natural-key application), `DECISIONS.md#017` (the once-per-source-finding semantics natural-key enforces), `DECISIONS.md#023` (publish's `decision_content_hash` consumer-side dedup pattern, the parallel mode-1 idiom).

## 027. V1 per-review publish-side advisory lock for concurrent-resume defense

**Status:** Accepted, 2026-05-26. Amended 2026-05-27 (lock-key derivation rewritten to a 64-bit slice of `review_id.bytes`; see Amended block below).

**Amended 2026-05-27:** the lock-key derivation changes from `pg_try_advisory_xact_lock(hashtext('publish:<uuid>'))` to `pg_try_advisory_xact_lock(<lock_id>)` where `lock_id = int.from_bytes(review_id.bytes[:8], byteorder="big", signed=True)`. Reason: `hashtext` returns int4 (32-bit), and the birthday bound on uniform 32-bit hashes is ~65k inputs before a 50% collision probability — at any realistic review volume distinct reviews would falsely serialize on the lock (wait up to `max_wait_seconds=120`; some publishes time out). Taking the first 8 bytes of `review_id.bytes` as a signed int8 drops collision probability to ~zero at any realistic volume and shares the int8 advisory-lock namespace with `SWEEP_LOCK_ID=0x4F55545244520001` without practical collision (uniform UUID distribution against the fixed sweep id). All three of the original alternative-rejection arguments (plain blocking; single-shot try-lock; bounded backoff) still apply — only the key derivation changes. The "No migration required" consequence below is rewritten to drop the 32-bit collision rationale and replace it with the int8-derivation rationale; the "Namespacing" paragraph is rewritten to describe the new namespace story; lines 948 and 956's body wording is updated to the new SQL shape. Sibling docstring/comment references in `src/outrider/audit/persister.py::acquire_publish_lock`, `src/outrider/audit/sinks.py::PublishEventSink.acquire_publish_lock`, `src/outrider/agent/nodes/publish.py`, `tests/integration/test_publish_lock_contention.py`, and `specs/2026-05-26-hitl-node.md` are updated in lockstep with this amendment. No change to the lock's behavior model — try-lock + bounded backoff, serialize-then-observe, deadline-bounded with `AuditPersisterPublishLockAcquisitionTimeoutError` — or to any other point in this decision.

**Context.** Until HITL shipped, the only V1 race against same-`review_id` duplicate publishes was webhook redispatch — covered by `DECISIONS.md#020`'s receiver-side idempotency check on `(repo_id, pr_number, head_sha)`. `FOLLOWUPS.md#FUP-068` named two attack vectors that the receiver check does NOT cover: (Attack 1) a forged `PublishEvent` row in `audit_events` — defense is HMAC provenance, V2 scope; (Attack 2) a TOCTOU race where two concurrent dispatcher invocations for the same `review_id` both observe `prior_publish_event=None` at `agent/nodes/publish.py` Step 4 and both POST to GitHub. FUP-068 originally scoped Attack 2 as V2 because V1's `BackgroundTasksDispatcher` is in-process and the V1 webhook-idempotency check closes the receiver-side window.

The HITL arc breaks that V1 framing. `POST /reviews/{review_id}/decide` enqueues `graph.ainvoke(Command(resume=...))` as a FastAPI `BackgroundTask`. `sweep/hitl_expiry.py::reclaim_stuck_hitl_states` ALSO invokes `compiled_graph.ainvoke(Command(resume=canonical_decision.model_dump(mode="json")))` for window-(f) recovery — same `review_id`, same in-process dispatcher, DIFFERENT background task. Both code paths can run concurrently against a single review; LangGraph per-thread checkpointer serialization is not a guarantee under concurrent `ainvoke` on the same `thread_id` (LangGraph 1.1.6). The race lands inside the publish node's Step 4 → Step 8 critical section: both invocations can observe `prior_publish_event=None` and both POST. Two GitHub reviews land for one logical decision. The receiver-side webhook idempotency check is irrelevant — neither path hits the receiver.

The fix the FUP named for V2 (option-b: per-`review_id` advisory lock bracketing publish-node Steps 4-8) is exactly the right shape, and the HITL surface forces it into V1.

**Decision.** V1 ships a per-review publish-side advisory lock via `PublishEventSink.acquire_publish_lock(review_id: UUID) -> AbstractAsyncContextManager[None]`, durable impl on `AuditPersister` backed by Postgres' `pg_try_advisory_xact_lock(<lock_id>)` where `lock_id = int.from_bytes(review_id.bytes[:8], byteorder="big", signed=True)`. The publish node enters the lock through `AsyncExitStack.enter_async_context(...)` BEFORE Step 4 and releases via `lock_stack.aclose()` at the function-finally boundary.

The lock acquisition is **try-lock with bounded backoff**, NOT plain blocking, NOT single-shot try-lock:

1. **Plain blocking `pg_advisory_xact_lock` was rejected.** A blocking variant holds the probe session's connection for the entire wait. With N same-review contenders, blocking pins N pool connections simultaneously — the winner's `emit_publish_routing` / `emit_publish_eligibility` / `emit_publish_attempt` / `emit_publish_result` calls (each opens a fresh `AsyncSession` per the per-emit session discipline) can be STARVED of pool connections by the held waiters plus whatever other workload (sweep jobs, dashboard reads, other publish paths) is running. This is a starvation pattern, not a strict deadlock — Postgres advisory locks themselves don't deadlock here because the holder is always making progress; the symptom is the winner blocking on pool acquisition while waiters hold connections idle, with the severity proportional to (pool size, concurrent same-review contender count, other workload). The conflict between "lock-holding session held across the GitHub round-trip" and "fresh session per emit_*" makes the starvation risk real enough to design against, even without a concrete pool-exhaustion proof.

2. **Single-shot `pg_try_advisory_xact_lock` with immediate loser-skip was rejected.** The immediate loser cannot distinguish "winner succeeded and committed `PublishEvent`" from "winner crashed between lock acquire and POST." A loser that short-circuits to `PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED` without observing the winner's outcome emits a false skip whenever the winner crashes mid-POST — and the publish is silently lost. The audit row's absence is the correct authority for "did publish actually happen"; a lock-release signal is not.

3. **Try-lock + bounded backoff is the accepted shape.** Each probe opens a fresh session+transaction, runs `pg_try_advisory_xact_lock(<lock_id>)` (where `lock_id` is the first 8 bytes of `review_id.bytes` as a signed int8 per Amended 2026-05-27 above), and on not-acquired RELEASES the session immediately (transaction rolls back, connection returned to pool) before sleeping with exponential backoff (50ms doubling to 1s cap). On acquired, holds the session+transaction for the lifetime of the `yield`. The eventual acquire puts the second task BEHIND the first's transaction boundary; Step 4's `query_prior_publish_event` then observes the first's committed `PublishEvent` (success → authentic `IDEMPOTENTLY_SKIPPED`) OR its absence (first crashed → second POSTs). False-skip eliminated by construction; pool pressure under contention drops from N held to ~1 held + occasional probes.

**Deadline + observability.** Default `max_wait_seconds=120` covers the realistic publish wall-clock (1-30s GitHub POST plus N-comment writes) with headroom. Exhaustion raises `AuditPersisterPublishLockAcquisitionTimeoutError` (carries `review_id` + `waited_seconds` as instance attributes; message contains only class-level identifiers per the `METADATA_ONLY_EXCEPTION_TYPES` contract). The publish node's outer try/except wrapping `enter_async_context(...)` emits `PublishAttemptEvent(outcome=failed, failure_class="AuditPersisterPublishLockAcquisitionTimeoutError")` BEFORE re-raising, honoring `agent/nodes/publish.py`'s raises contract. Lock-acquisition I/O failure (DB outage, connection drop) raises through the same outer catch. The structural split — lock-acquire in its own try, critical-section in a separate try/finally calling `lock_stack.aclose()` — means inner-step failures (Step 4 / Step 6 / Step 7) reach only their own existing FAILED emits, never the outer catch, so no double-emit.

**Namespacing.** The advisory-lock key is the first 8 bytes of `review_id.bytes` interpreted as a signed int8 (per Amended 2026-05-27 above), computed Python-side and passed as the bigint argument to `pg_try_advisory_xact_lock`. It does NOT collide in practice with `SWEEP_LOCK_ID=0x4F55545244520001` (the fixed bigint at `sweep/purge_expired.py::SWEEP_LOCK_ID`): UUIDs span ~2^128 distinct values and the 64-bit slice has ~2^64 distinct images, so a uniform-distribution collision against a fixed sweep id is negligible. When `sweep/hitl_expiry.py::reclaim_stuck_hitl_states` drives `Command(resume=...)` through the graph (executing under the outer `SWEEP_LOCK_ID` transaction), the publish node's `acquire_publish_lock` opens a SEPARATE session and acquires the per-review int8 lock — different keys, no deadlock.

**Consequences.**

- **FUP-068 Attack 2 closed in V1** (defense-in-depth WITH webhook-idempotency, not instead of it). FUP-068 Attack 1 (forged-`PublishEvent` denial-of-publish) remains the open V2 work; the lock partially mitigates it (a forged-row attack now requires perfect timing against the lock).
- **Lock boundary: Postgres-mediated, not in-process.** Advisory locks live in Postgres and coordinate across ALL processes sharing the same database — V1's `BackgroundTasksDispatcher` (in-process), a hypothetical second app server pointing at the same DB, and a V2 `CeleryDispatcher` (multi-process) all serialize correctly through this lock. The boundary the lock CANNOT cross is the GitHub side effect itself: the lock auto-releases when the holder's transaction closes (commit, rollback, connection drop, process crash, `pg_terminate_backend`), and that release signals nothing about whether the in-flight GitHub POST actually landed. A process that crashed AFTER the GitHub POST committed server-side but BEFORE `emit_publish_result` persisted the local `PublishEvent` row leaves the lock released, no audit row, and a real GitHub review on the PR. `agent/nodes/publish.py` Step 6's `find_existing_review_on_head_sha` body-marker check is the defense for that scenario — matches by `<!-- outrider-review-id:{uuid} -->` body marker on a process restart, when the local audit row's absence is misleading.
- **New persister method + new exception type.** `AuditPersister` gains `acquire_publish_lock` (the eighth method on the durable class) and `AuditPersisterPublishLockAcquisitionTimeoutError` (enrolled in `METADATA_ONLY_EXCEPTION_TYPES` per the `DECISIONS.md#016` metadata-only contract). The Protocol surface `PublishEventSink` gains one method; the `audit/sinks.py::PublishEventSink` `dir(...)` membership pin in `tests/unit/test_github_publisher.py` is updated; seven existing test sinks (`tests/unit/test_publish_idempotency.py`, `tests/unit/test_publish_node_end_to_end.py`, `tests/unit/test_publish_routing.py`, `tests/unit/test_graph_skip_routing.py`, `tests/unit/test_agent_graph_builder.py`, `tests/integration/test_review_state_langgraph_merge.py`, `tests/integration/test_analyze_graph_wiring.py`) carry no-op `acquire_publish_lock` implementations.
- **`reclaim_stuck_hitl_states` candidate query broadened.** The window-(f) recovery race that motivated this entry surfaced a sibling drift in the sweep predicate: the original `WHERE status='awaiting_approval' AND expires_at < grace_cutoff` filter would miss rows that the `transition_expired_hitl_reviews` sub-job had ALREADY flipped to `awaiting_approval_expired` at `expires_at < now`. The broadened predicate — `status IN ('awaiting_approval', 'awaiting_approval_expired')` with NO `expires_at` filter and audit-row check FIRST (no grace gate; grace gate applies only to the no-audit window-c path) — is the safety net that keeps window-(f) orphans reachable regardless of sub-job-ordering races. This is a sibling fold landed in the same commit; it does not on its own warrant a DECISIONS entry but is named here as a load-bearing context for the publish-lock's race story.
- **Audit-event-append-only invariant preserved.** The lock is an advisory lock, not an audit-events write. `pg_try_advisory_xact_lock` returns a boolean and acquires a session-scoped lock; no INSERT, UPDATE, or DELETE against `audit_events` happens inside the lock-acquire path. The `audit-events-append-only` invariant + `audit_events_no_mutation` DDL trigger from `db/triggers/audit_append_only.sql` are not touched.
- **No migration required.** Postgres' `pg_try_advisory_xact_lock` is a built-in primitive; no schema or extension changes. The lock namespace is per-cluster (advisory locks share a 64-bit integer namespace across all databases on the cluster). Per Amended 2026-05-27 above, the key is computed Python-side as `int.from_bytes(review_id.bytes[:8], byteorder="big", signed=True)` and passed as the bigint argument to the single-argument `pg_try_advisory_xact_lock(bigint)` form — no Postgres-side hashing. Collision probability at realistic review volume is negligible: UUIDs distribute ~uniformly over 2^128 values, the 64-bit slice over ~2^64 values; the birthday bound on uniform 64-bit keys is ~2^32 (~4B) inputs before a 50% collision probability, which is many orders of magnitude beyond V1 contention. If two distinct `(review_id_a, review_id_b)` did collide, the consequence is unnecessary serialization (one waits for the other's transaction to commit; both eventually proceed correctly), NOT a safety or correctness violation — the post-lock `query_prior_publish_event` reads use the actual `review_id`, not the lock_id. Cross-application collision with other tools sharing the cluster is the same shape: independent applications' advisory locks contend on the SAME 64-bit space; the UUID-derived key uses bits the application chooses, so accidental collisions with other tools' fixed-id schemes (like Outrider's own `SWEEP_LOCK_ID=0x4F55545244520001`) are uniform-random over the int8 space. No other Outrider component contends on derived advisory locks (the sweep family uses the fixed `SWEEP_LOCK_ID` bigint).
- **Operator visibility.** `AuditPersisterPublishLockAcquisitionTimeoutError` rendering carries `review_id` only in the message string (`waited_seconds` lives on the instance attribute, not the rendered str, per the `METADATA_ONLY_EXCEPTION_TYPES` contract under stringified annotations). Operators investigating a timeout read the exception class name, the `review_id`, and the underlying `PublishAttemptEvent(outcome=failed, failure_class="AuditPersisterPublishLockAcquisitionTimeoutError")` row to correlate.

**Why this is a DECISIONS entry, not just inline comments.** Three reasons satisfy the `docs/workflow.md` DECISIONS-tier criteria:

1. **Likely cited from code comments later** — the rationale (why bounded try-lock over blocking, why bounded retry over single-shot loser-skip) is exactly the kind of "why is this here" question a future reader will ask. The persister + publish-node comments already cite the pattern; they should cite the anchor.
2. **Stable, not imminently revisable** — the design has been through three audit rounds (initial F2 fold, F5 false-skip critique, F8 connection-pool starvation critique); each fold tightened it, and the current shape passes all three constraints. V2 hardening adds Attack-1 provenance (FUP-068) but doesn't change the lock mechanism.
3. **Context unreconstructable from code alone** — the rejected alternatives (plain blocking, single-shot try-lock) are NOT visible in the current code. A future reader without this entry would not know why try-lock + backoff was chosen over the simpler alternatives, and could "simplify" the code into a bug.

**Referenced from.** `src/outrider/audit/persister.py::acquire_publish_lock` (durable impl + rejected-alternative rationale in docstring), `src/outrider/audit/persister.py::AuditPersisterPublishLockAcquisitionTimeoutError` (exception class + enrolled in `METADATA_ONLY_EXCEPTION_TYPES`), `src/outrider/audit/sinks.py::PublishEventSink.acquire_publish_lock` (Protocol surface), `src/outrider/agent/nodes/publish.py` (call site at the lock-acquire block; outer try/except for FAILED emit on acquire failure), `tests/integration/test_publish_lock_contention.py` (executable contract pin — serialize-then-observe + cross-review-independence + timeout-raises), `tests/integration/test_anomaly_persister_duplicate.py` (sibling integration test on the partial-unique-index pattern this entry's `index_elements`+`index_where` discipline shares with), `FOLLOWUPS.md#FUP-068` (precursor — Attack 2 closed in V1 per this entry; Attack 1 remains V2 work), `tests/unit/test_github_publisher.py` (Protocol-method-set pin including `acquire_publish_lock`). Spec amendment to `docs/spec.md` §V (publish-node design narrative) is deferred to a separate commit per the workflow's docs-only commit discipline.

## 028. Per-review policy_version snapshot anchor on TriageResult; triage gate enforces producer-side integrity (V1 scope)

**Status:** Accepted, 2026-05-28.

**Context.** Synthesize's H-1 forge defense needs a trusted per-review `policy_version` snapshot so that (a) a single review's findings, summary, and replay share one policy version regardless of mid-deploy `ACTIVE_POLICY_VERSION` bumps, and (b) a downstream attacker-influenced producer (analyze proposals, prompt-injection from PR content) cannot poison the anchor by planting a forged finding with a divergent `policy_version` value.

Two earlier resolutions were tried during the synthesize-node audit-the-audit loop and rejected:

- **Direct comparison against the live `ACTIVE_POLICY_VERSION` constant** (Pass-2). The constant is uninfluenceable but fails under mid-deploy hot-reload: a review whose triage classified findings under version 0.0.0 then hits synthesize after the constant bumps to 0.0.1 — every legitimate finding now diverges from live and synthesize denies completion. Operational hazard, not a security one, but a release blocker.

- **First-finding-anchored snapshot** (Pass-3). Capturing the anchor from `analysis_rounds[0].findings[0].policy_version` survives the bump but moves the trust root into attacker-influenceable space: a single forged finding planted in round 0 index 0 poisons the snapshot for every subsequent legitimate finding, denying completion via single-finding-poisoning DoS.

Remaining design space — move the snapshot UPSTREAM of analyze (only triage is upstream and runs before any attacker-influenced LLM call other than its own — which itself can be gated), and add a producer-side gate at triage that rejects any LLM-injected divergent `policy_version` (an LLM emitting `{"policy_version": "0.0.0", ...}` in its triage JSON survives Pydantic's `pattern=BARE_SEMVER_PATTERN` shape floor because the field admits any valid semver).

**Decision.** #028 establishes the snapshot anchor surface and the producer-side triage gate; the downstream-consumer story for analyze in cross-process durable-retry deployments is the subject of #029.

1. `TriageResult` gains a `policy_version: str = Field(default_factory=lambda: ACTIVE_POLICY_VERSION, pattern=BARE_SEMVER_PATTERN)` field. `default_factory` fires on field omission (the canonical happy path — triage LLM should not emit this field); the pattern is the schema-level shape floor.

2. `agent/nodes/triage.py::_enforce_triage_policy` gains Rule (d): post-validation, raise `TriagePolicyViolationError` if `result.policy_version != ACTIVE_POLICY_VERSION`. The rule rejects DIVERGENT values; exact-active injection is acceptable (an LLM that injects exactly the active version produces an audit-row indistinguishable from the legitimate default_factory path, and the attacker gains nothing). If exact-active injection must ALSO be impossible — i.e., the field should never appear in triage LLM output at all — the right fix is a separate LLM-only DTO that excludes the field, OR a pre-validation reject of raw JSON containing the key. V1 ships value-integrity rejection only.

3. `agent/nodes/synthesize.py::_enforce_synthesize_input_invariants` compares every finding's `policy_version` against `state.triage_result.policy_version` (the triage-captured snapshot) and raises `FindingForgeryDetectedError` on divergence.

4. `SynthesizeCompletedEvent.policy_version` mirrors `state.triage_result.policy_version` so the audit row records the snapshot under which findings were classified. `PublishEligibilityEvent.policy_version` mirrors `finding.policy_version` (same snapshot, propagated through the finding).

5. The replay path is exempt from Rule (d). Replay reconstruction reads `audit_events` rows directly and does NOT re-execute the triage node, so a historical `policy_version` rehydrated from an old TriageResult stays valid for the replay.

**V1 scope limitation (explicit, not an oversight).** Analyze stamps findings with its own `active_policy_version` parameter (defaulting to live `ACTIVE_POLICY_VERSION` at module-import time), NOT with `state.triage_result.policy_version`. Within V1's in-process `BackgroundTasksDispatcher`, this is safe: `ACTIVE_POLICY_VERSION` is `Final` per process lifetime; no mid-process bump. A mid-deploy bump kills the in-flight review via process death (BackgroundTasks doesn't cross process boundaries), so the snapshot mismatch between `finding.policy_version` (live) and `triage.policy_version` (snapshot) doesn't fire on resume. The trust root is complete under V1's in-process dispatcher assumptions. **V2 Celery durable-retry crosses processes** and will require analyze to consume the triage snapshot at runtime; see #029.

**Consequences.**

- `TriageResult` deviates from `docs/spec.md` §7.2's four-field shape (adds a fifth field). The canonical record (spec.md §7.2) carries an `Amended 2026-05-28 — see DECISIONS.md#028` note referencing this decision; the field is documented as the snapshot-anchor surface for synthesize's H-1 defense.

- Triage's deterministic policy gate (`_enforce_triage_policy`) acquires a fourth rule beyond the existing path-set rules (a/b/c). The new rule shares the same exception class (`TriagePolicyViolationError`) and bounded-sample log-content discipline.

- A future "explicit historical-version triage" code path (e.g., a tool that constructs a TriageResult from an old audit row to feed a what-if analysis) must bypass Rule (d) by NOT going through `_enforce_triage_policy`. The replay path already satisfies this by not calling the triage node at all.

- The schema-level `pattern=BARE_SEMVER_PATTERN` floor is retained even though Rule (d) is a strictly tighter gate at the only production write site. The pattern catches malformed `policy_version` values that arrive through other construction paths (rehydration from a corrupted audit row, future test fixture mistakes); removing it would narrow the safety net without operational benefit.

- The Pydantic `default_factory` semantics are load-bearing: it fires on field omission via `model_validate_json` (verified against `pydantic/concepts/fields/index.md` during the synthesize-node audit-the-audit). A future Pydantic upgrade that changes this contract would silently break the happy path; pin the version + add an integration test that asserts the factory fires on missing-field JSON.

- The V1 trust root is complete within the deployment model V1 ships (in-process BackgroundTasks). #029 extends the trust root to cross-process durable retry before V2 Celery + Redis lands.

**Referenced from.** `src/outrider/schemas/triage_result.py` (the `policy_version` field), `src/outrider/agent/nodes/triage.py` (`_enforce_triage_policy` Rule (d) + `TriagePolicyViolationError` enrollment), `src/outrider/agent/nodes/synthesize.py` (`_enforce_synthesize_input_invariants` triage-anchored comparison + `FindingForgeryDetectedError`), `src/outrider/audit/events.py` (`SynthesizeCompletedEvent.policy_version` snapshot mirror + `PublishEligibilityEvent.policy_version` snapshot mirror), `src/outrider/agent/nodes/publish.py` (`PublishEligibilityEvent` emit site stamping `finding.policy_version`), `tests/unit/test_triage_node.py` (`test_enforce_policy_rejects_llm_injected_policy_version` pin), `tests/unit/test_triage_result.py` (`test_triage_result_policy_version_admits_any_valid_semver` shape-vs-value pin), `tests/unit/test_synthesize_node_defenses.py` (`test_synthesize_blocks_first_finding_poisoning` triage-anchored snapshot defense pin), `tests/unit/test_publish_routing.py` (`test_publish_eligibility_stamps_finding_policy_version_not_active` snapshot mirror pin).

## 029. Cross-process durable-retry policy snapshot consumption: analyze consumes triage snapshot via closure-injected loader; synthesize retry idempotency via DB-arbitrated reconstruction

**Status:** Drafted, 2026-05-28. Trigger-gated on V2 Celery + Redis landing — points 1-8 below MUST land before the dispatcher swap ships.

**Context.** #028 establishes the per-review policy_version snapshot anchor on `TriageResult` and the producer-side triage gate. The trust root is complete under V1's in-process dispatcher assumptions: `Final ACTIVE_POLICY_VERSION` per process lifetime; mid-deploy bumps kill in-flight reviews via process death.

V2 changes the deployment model. `CeleryDispatcher` provides durable retry — a task that crashes mid-execution requeues and runs in a new worker process whose `ACTIVE_POLICY_VERSION` and `SEVERITY_POLICY` mapping may have bumped since the original triage. Three coupled break modes emerge:

1. **Analyze-side label drift.** `analyze.py::analyze` defaults `active_policy_version` to live `ACTIVE_POLICY_VERSION`; the parser stamps that value on every finding. Cross-process retry stamps v2 onto findings from a v1-anchored review. Synthesize compares against `triage_result.policy_version` (v1) → divergence raises.

2. **Analyze-side mapping drift.** Parser reads live `SEVERITY_POLICY[finding_type]` synchronously at classification time. If the mapping (not just the version label) bumps, the row's (label, severity) pair is incoherent. Replay reconstruction asserting severity matches the labelled mapping fails.

3. **Synthesize retry state-audit lockstep + concurrent-retry race.** Two failure shapes that compound:
   - **Sequential crash recovery:** crash after `SynthesizeCompletedEvent` emit + before node return → retry makes fresh LLM call → fresh summary text → returned `ReviewReport`'s `summary_content_hash` ≠ persisted event's. `DECISIONS.md#026` gate iii (state-lockstep) violated.
   - **Concurrent retry race** (V2 Celery dispatcher may double-fire under network-partition / lost-ack): two synthesize invocations both observe no prior `SynthesizeCompletedEvent`, both call the LLM, both emit fresh event-id-PK rows. Pre-flight check alone is insufficient; same race class as `DECISIONS.md#027`'s publish-side advisory lock concern.

These three break modes are coupled: all three fire when V2 lands without further work. Bundling them avoids a partial-V2-trust-root intermediate state.

**Decision.** Land the following before V2 Celery + Redis enters the hot path. The nine points fall into three groups: items 1-3 are analyze-side (close break modes 1 + 2); items 4-8 are synthesize-side (close break mode 3, both layers required — pre-flight as fast path + DB arbitration as correctness path); item 9 is the shared rehydration test gate.

1. **Analyze fails loud on missing `triage_result`.** `analyze.py::analyze` asserts at entry; raises `AnalyzeMissingTriageError` (new typed exception, sibling of synthesize-side `FindingForgeryDetectedError`). The current Pass-3 fallback that treats missing triage as all-files-SKIP becomes wrong under #029's trust-root model — a silent all-SKIP review under the wrong snapshot leaves a poisoned audit trail.

2. **Analyze reads `policy_version` from state at runtime.** Drop `active_policy_version: str = ACTIVE_POLICY_VERSION` from the node-entrypoint `analyze.py::analyze` signature. Body reads `policy_version = state.triage_result.policy_version` immediately after the fail-loud assertion. Internal helpers (`_process_one_file`, `parse_analyze_response`, etc.) continue to accept `policy_version: str` AND newly accept a `severity_policy: Mapping[FindingType, FindingSeverity]` kwarg. The change is scoped to the entrypoint signature + the new mapping parameter that joins it; no internal helper is renamed.

3. **Closure-injected versioned-policy loader.** `build_graph` injects a versioned policy loader dep into `analyze` per `nodes-receive-deps-via-closure`. The loader is async (matches the existing `policy/versions.py::load_policy_for_version(version, conn: AsyncConnection)` shape). A closure-owned async-aware cache keyed by version string returns the resolved immutable mapping to callers; `functools.lru_cache` is NOT correct here (it would cache the coroutine object, not its awaited result, and awaiting the same coroutine twice raises `RuntimeError`). Implementation options: plain `dict[str, Mapping[FindingType, FindingSeverity]]` populated on first await (single-writer per process is the V2 dispatcher contract); `asyncio.Lock` + dict if concurrent awaits within one process become possible; or a third-party `async-lru` if its availability is acceptable. The decision is SHAPE not implementation: one async resolution per (process, version), result is immutable, passed sync to the parser. Analyze awaits the loader ONCE per node invocation, then passes the resolved `severity_policy: Mapping[FindingType, FindingSeverity]` into the sync parser path as plain data. The parser stays sync; async I/O stays on the node.

4. **Pre-flight reconstruction (fast path).** *(Synthesize-side; closes break mode 3. Both layers — pre-flight + DB arbitration — are required.)* Synthesize body queries `audit_events` for an existing `SynthesizeCompletedEvent` keyed on `(review_id, event_type='synthesize_completed')` at step 1 (before any LLM work). If found → reconstruct (step 7 below) and return. If not found → proceed with normal flow. This is an optimization, NOT the correctness mechanism; it short-circuits the LLM call cost when retry obviously fires.

5. **DB-arbitrated dedup (correctness path).** A new partial unique index lands on `audit_events`, keyed on the normalized top-level `event_type` column (NOT a JSONB payload lookup): `CREATE UNIQUE INDEX synthesize_completed_per_review_idx ON audit_events (review_id, event_type) WHERE event_type = 'synthesize_completed'`. Same pattern as the anomaly-subsystem per-rule partial unique indexes. At emit time, the persister uses `postgresql_insert(...).on_conflict_do_nothing(...).returning(event_id)`. If the conflict path fires (concurrent second invocation arrived first), `RETURNING` yields no row → the caller (synthesize body) discards the fresh LLM response, queries the persisted winning row, reconstructs from it, and returns the reconstructed state delta. This satisfies `DECISIONS.md#026` gate iii because the returned state IS the persisted state, not a regenerated one.

6. **Deterministic content-join binding via `llm_call_event_id`.** Today's `LLMResponse` (`llm/base.py::LLMResponse` class at line 577 area; the `text` field is the relevant existing surface) does not expose the persisted `LLMCallEvent.event_id` — the join from `SynthesizeCompletedEvent` back to `llm_call_content` is under-specified; retry/crash paths can leave multiple synthesize LLM call rows in the audit log, making `(review_id, node_id='synthesize')` cardinality > 1. Two changes close this:

   - `LLMResponse` gains a new required field `llm_call_event_id: UUID` (the event_id the provider persisted alongside the response in the same transaction). All Protocol implementations populate it; the field is non-Optional to prevent silent drift.

   - `SynthesizeCompletedEvent` gains `llm_call_event_id: UUID | None` (Optional, NOT required). Carried verbatim from the `LLMResponse` for any row emitted at #029 deployment or later. The field is Optional because pre-#029 V1 production rows exist (synthesize-node ships in V1; #029 triggers when V2 work begins, by which time historical V1 rows are in `audit_events`). `audit-events-append-only` prohibits backfilling — historical rows stay `llm_call_event_id=None` forever; Pydantic discriminator parses them via the Optional path.

   Reconstruction path branches on the field's presence:

   - **Post-#029 row** (`llm_call_event_id is not None`): direct join on `llm_call_content.event_id = synthesize_event.llm_call_event_id` — exactly-one cardinality by schema (LLMCallEvent event_ids are append-only PK unique).
   - **Pre-#029 row** (`llm_call_event_id is None`): fallback join on `(review_id, node_id='synthesize')` with explicit cardinality check — if rows returned == 1, reconstruct; if > 1 (the under-specified case that motivated this fix in the first place), raise `SynthesizeLegacyAmbiguousJoinError`. The fallback is the only correct path for V1 audit rows; #029 doesn't backfill, doesn't mutate, doesn't pretend the legacy rows are V2-shape.

   The alternative (join by `summary_content_hash` exact-match with a cardinality check) was considered but rejected: it requires a runtime cardinality assertion as the integrity gate, while the `llm_call_event_id` foreign-key-style binding pushes the integrity into the schema for new rows. Cardinality is structural for new rows, runtime-checked only for the legacy bridge.

7. **Full-aggregate state-audit lockstep validation.** `llm_call_event_id` fixes the summary-content join, but `DECISIONS.md#026` gate iii is full state-audit lockstep — not just summary. The reconstruction path validates EVERY field that appears on both the persisted `SynthesizeCompletedEvent` and the reconstructed `ReviewReport`:

   - `overall_risk` (event mirror of `ReviewReport.overall_risk`)
   - `n_findings` (event mirror of `len(ReviewReport.findings)`)
   - `policy_version` (event mirror of triage snapshot)
   - `synthesize_model` (event mirror of the LLM model that produced the summary)
   - `summary_content_hash` (event mirror of sha256(raw response.text))
   - Metrics aggregates if they're on the event (`files_examined`, `wall_clock_seconds`, etc. — whichever the event carries; metrics not on the event are reconstructed from state and not validated).

   On any drift, raise `SynthesizeReconstructionMismatchError(field=<which>, persisted=<value>, reconstructed=<value>)`. Drift is a bug, not a recoverable retry state — it means the rehydrated `state.analysis_rounds` or `state.triage_result` itself drifted between original emit and retry, which is a deeper integrity failure than the dedup mechanism is designed to absorb.

   For finding-level identity (the deepest lockstep), the event carries `n_findings` only; per-finding content_hashes would require either expanding the event payload (rejected per #026's metadata-only stance) OR joining `audit_events` on individual `FindingEvent` rows (already emitted per finding under `audit-events-append-only`). The reconstruction path uses the latter, scoped to what the current audit schema actually carries: `FindingEvent` rows do NOT carry a `round_index` / `pass_index` field (verified against `audit/events.py::FindingEvent` at #029 draft time), so per-finding validation is review-scoped, not round-scoped. Query `FindingEvent` rows for `(review_id,)`, extract the distinct set of `finding_content_hash` values, and compare against the set of `content_hash` values on reconstructed `ReviewReport.findings`. Set comparison (NOT sequence comparison) is the correct shape because:

   - `content_hash` is the synthesize-spec dedup key — duplicate content_hashes across rounds collapse to one entry in `ReviewReport.findings` per `ReviewReport._canonicalize_findings`, and the set comparison absorbs that collapse symmetrically.
   - Cross-round emission of the same finding (analyze pass 1 + pass 2 both surfacing the same proposal) lands as multiple `FindingEvent` rows with identical `finding_content_hash`; the DISTINCT projection on the audit-query side mirrors the synthesize-side dedup. Without DISTINCT, the cardinalities would diverge structurally and the validation would false-positive on every multi-round review.
   - The event's `n_findings` aggregate is validated separately (against `len(reconstructed.findings)`); together with the set-equality check, this catches both "extra finding in reconstructed but not persisted" and "missing finding in reconstructed".

   This brings the lockstep to per-finding identity at review granularity without expanding the `SynthesizeCompletedEvent` payload. If a future need surfaces for round-scoped validation (e.g., replay reconstructs `analysis_rounds[0]` vs `analysis_rounds[1]` independently and must validate each round's findings separately), a separate decision adds a `round_index: int` field to `FindingEvent`. That binding does NOT exist today and #029 does NOT add it; #029's lockstep is review-scoped because the audit surface is review-scoped.

8. **Retry-after-TTL fail-loud.** Reconstruction depends on `llm_call_content` being within retention. V2 durable retry should fire within the dispatcher's retry-policy timeout (seconds to minutes), well within LLM-content TTL. If the dispatcher EVER retries after TTL (not the V2 design intent), the persisted event exists but the joined content row is gone. Fail-loud with `SynthesizeRetryAfterContentPurgedError` rather than silently regenerating: regeneration would silently break state-audit lockstep against the prior event.

9. **Checkpoint-rehydration integration test.** *(Shared; applies to both analyze-side and synthesize-side correctness.)* Verify via integration test that rehydrating a `ReviewState` from a LangGraph checkpoint preserves `triage_result.policy_version` exactly (no `default_factory` fire on the rehydrated path; the field carries its captured snapshot verbatim). Load-bearing Pydantic contract under V2; regression backstop against a future Pydantic upgrade that changes default_factory semantics.

Items 1-3 close break modes 1 and 2 (analyze-side label / mapping drift). Items 4-8 close break mode 3 (synthesize-side state-audit lockstep + concurrent-retry race). Item 9 is the shared rehydration test gate.

**Out of scope.**

- LangGraph checkpoint summary-content retention (FUP-095). That's a retention-authority decision (does the checkpoint payload have its own TTL? does it mirror llm_call_content TTL?), not a policy-snapshot-consumption decision. Bundling would muddy the scope.
- Replacing `BackgroundTasksDispatcher` → `CeleryDispatcher` itself. The dispatcher swap is V2's spec concern; #029 covers the policy-snapshot + idempotency prerequisites that must land before the swap is safe.
- LLM-only DTO that excludes `policy_version` from triage's Pydantic admission surface (the stronger "LLM cannot mention hidden fields" boundary). V1+V2 ship value-integrity rejection only.

**Consequences.**

- Analyze's entry signature simplifies (one fewer parameter) but gains a closure-injected loader dependency. Tests that monkeypatched `active_policy_version=` lose the override hook; the new contract is "test sets `state.triage_result.policy_version` directly + stubs the loader to return the desired mapping."

- `load_policy_for_version` becomes a hot-path lookup. The closure-owned async-aware cache (keyed by version string) keeps it cheap — per-process amortization across reviews of the same version. The parser stays sync; no per-finding async hop.

- `LLMResponse` schema gains a non-Optional `llm_call_event_id: UUID`. All providers (`AnthropicProvider`, future `OpenAIProvider`) + the test `_MockLLMProvider` set it. Mocks that previously returned a hand-rolled `LLMResponse` need updating; the migration is mechanical (every emit site populates the field).

- `SynthesizeCompletedEvent.llm_call_event_id` is `UUID | None` (Optional). New rows (post-#029 deployment) populate it; legacy V1 rows (pre-#029) stay `None` forever per `audit-events-append-only`. Reconstruction code branches: direct join for new rows, cardinality-checked legacy join + fail-loud on ambiguity for legacy rows. No migration touches existing `audit_events` payloads.

- A new partial unique index lands on `audit_events`, keyed on the normalized top-level `event_type` column (NOT a JSONB payload lookup): `CREATE UNIQUE INDEX synthesize_completed_per_review_idx ON audit_events (review_id, event_type) WHERE event_type = 'synthesize_completed'`. Same shape as the anomaly-subsystem per-rule indexes; migration template is reusable.

- `_persist_non_phase_event` learns the same conflict-RETURNING pattern `emit_phase` already uses for natural-key dedup. The caller-side hook (returning None from `emit_synthesize_completed` means "row already exists; reconstruct") is the new Protocol contract — `SynthesizeEventSink.emit_synthesize_completed` return type changes from `None` to `EventEmitOutcome` (an enum with `INSERTED` / `EXISTING`). Sink Protocol versioning required.

- The pre-flight + DB-arbitration shape is sibling of `#027`'s publish-side advisory lock (same race class, different resolution mechanism — publish uses lock because it has side effects; synthesize uses unique index because the side effect IS the audit row).

- A future "reconstruct prior synthesize for an audit-replay test" utility reuses the same reconstruction logic (steps 4-7 above). Worth landing as a shared `audit/reconstruct_synthesize.py` helper so V2 retry path + audit-replay path share one canonical implementation.

- The `event_id-PK idempotency` claim in `SynthesizeCompletedEvent`'s class docstring stays correct for individual rows (each row's `event_id` is unique). What the docstring NEEDS to add is the per-review-aggregate semantics via the partial unique index — DB arbitration replaces the event_id-PK contract for per-review uniqueness. Update the docstring as part of the #029 implementation PR.

**Referenced from.** `src/outrider/agent/nodes/analyze.py` (entry signature change, fail-loud, runtime read from triage snapshot), `src/outrider/agent/nodes/synthesize.py` (pre-flight reconstruction + DB-arbitrated conflict path + full-aggregate validation), `src/outrider/llm/base.py` (`LLMResponse.llm_call_event_id`), `src/outrider/llm/anthropic_provider.py` (event_id population), `src/outrider/audit/events.py` (`SynthesizeCompletedEvent.llm_call_event_id` optional field + docstring amendment), `src/outrider/audit/sinks.py` (`SynthesizeEventSink.emit_synthesize_completed` return type changes to `EventEmitOutcome`), `src/outrider/audit/persister.py` (conflict-RETURNING + partial unique index handling), `src/outrider/audit/reconstruct_synthesize.py` (TBD — shared retry + replay helper), `src/outrider/policy/versions.py` (`load_policy_for_version` becomes hot-path), `src/outrider/agent/graph.py` (loader injection at `build_graph`), `db/migrations/versions/<TBD>_synthesize_completed_natural_key.py` (partial unique index migration), `tests/integration/test_synthesize_durable_retry_db_arbitration.py` (TBD), `tests/integration/test_analyze_consumes_triage_snapshot.py` (TBD), `tests/integration/test_review_state_rehydration_preserves_policy_version.py` (TBD).

## 030. ReviewReport tuple-not-list findings field (permanent) + V1-transition nullable ReviewMetrics LLM-aggregate fields (FUP-093 revert)

**Status:** Accepted, 2026-05-28.

**Context.** The synthesize-node implementation at `src/outrider/schemas/review_report.py` ships two deviations from `docs/spec.md` §7.3's ReviewReport / ReviewMetrics shape. Both surfaced during the synthesize-node audit-the-audit loop; neither is accidental, but both deserve canonical-record reconciliation rather than living as code-comment-only divergences (the failure mode `docs/conventions.md` "Spec fidelity" warns against — quiet drift between spec and code that the audit chain has to re-discover each pass).

Deviation 1 — `findings` container shape. Spec §7.3 declares `findings: list["ReviewFinding"]`. Implementation at `src/outrider/schemas/review_report.py:176` is `tuple[ReviewFinding, ...] = Field(max_length=200)`. `ReviewReport.model_config = ConfigDict(frozen=True, ...)` provides only shallow immutability — `frozen=True` on a model carrying a `list` field is faux-immutable over `.append()` / `.pop()` on the list itself, so the envelope's frozen contract leaks. The tuple choice plus the `_canonicalize_findings` field_validator (returns `tuple(sorted(...))` and rejects duplicate content_hashes) is the working pattern already established at three sibling sites: `PRContext.changed_files: tuple[ChangedFile, ...]` (schemas/pr_context.py), `HITLDecision.decisions: tuple[PerFindingDecision, ...]` (schemas/hitl.py, per `DECISIONS.md#014` Amended 2026-04-29), `AnalysisRound.findings: tuple[ReviewFinding, ...] = Field(max_length=50)` (schemas/analysis_round.py). Spec §7.3's `list[...]` annotation is unmaintained drift from before the tuple-precedent landed.

Deviation 2 — `ReviewMetrics` LLM-aggregate nullability. Spec §7.3 declares `llm_calls_made: int`, `total_input_tokens: int`, `total_output_tokens: int`, `total_cost_usd: float` — all required, non-optional. Implementation at `src/outrider/schemas/review_report.py:130-139` is `int | None = Field(default=None, ge=0)` (token + call-count fields) and `float | None = Field(default=None, ge=0, le=100.0)` (cost field). The other `ReviewMetrics` fields (`files_examined`, `files_traced_beyond_diff`, `wall_clock_seconds`) match the spec — they are deterministically computable from `state.analysis_rounds` / `state.trace_decisions` (union with `state.trace_fetched_files` for `files_traced_beyond_diff` per `_compute_files_traced_beyond_diff` — "beyond diff = outside changed-files set," not "Phase-2-fetched specifically") / node-side `time.monotonic()` delta. The four LLM-aggregate fields are NOT yet computable: the audit-query helper that would sum `LLMCallEvent.input_tokens` / `output_tokens` / `cost_usd` joined on `review_id` is not yet wired (`FOLLOWUPS.md#FUP-093`). Two V1 shapes were considered:

- Placeholder-zero (`int = Field(default=0)`). Rejected: emits durable false metadata on every V1 audit row — the row claims "this review made 0 LLM calls" when the real answer is unknown. Replay reads that row at face value; the dashboard reads that row at face value; downstream cost-budget alerting would treat zero as a green signal. False-zero is worse than absent.
- Nullable + None default (`int | None = Field(default=None, ge=0)`). Distinguishes "unknown vs. zero" honestly on the audit row. Dashboard reads audit-events truth directly (`LLMCallEvent` rows) for these aggregates; the denormalized `ReviewMetrics` snapshot is a convenience, not load-bearing.

V1 ships the nullable shape. When FUP-093 lands (audit-query helper wired into `synthesize._compute_metrics`), the fields become required `int` / `float` and the spec returns to the canonical shape — this deviation is a documented transition state, NOT a permanent canonical change.

**Decision.** Both deviations are accepted with asymmetric durability:

1. `ReviewReport.findings: tuple[ReviewFinding, ...] = Field(max_length=200)` is the **permanent** canonical shape. The spec.md §7.3 inline `list[ReviewFinding]` annotation is amended to `tuple[ReviewFinding, ...]` with a compact `# See DECISIONS.md#030` pointer (rationale lives here, NOT inline in the spec). The `_canonicalize_findings` validator (severity-then-location sort + duplicate-content_hash rejection) is the matching pattern, mirroring `TriageResult._canonicalize_relevant_dimensions`.

2. `ReviewMetrics.llm_calls_made` / `total_input_tokens` / `total_output_tokens` / `total_cost_usd` are V1 **transition-state nullable**. Spec.md §7.3 carries a compact `# See DECISIONS.md#030` pointer next to each of the four fields. The schema-field default is `None` (not `0`); `ge=0` / `le=100.0` floors and caps apply when the value is non-None. Forward-compat: when FUP-093 closes (audit-query helper landed, fields populated), the field annotations drop `| None`, the `default=None` clauses drop, and the spec-amendment pointers delete — the canonical record returns to the spec's original required-int / required-float shape.

3. The `_enforce_synthesize_input_invariants` H-1 forge defense (per `DECISIONS.md#028`) and the `ReviewReport`-level frozen contract apply uniformly across both deviations — no security or replay-equivalence regression is introduced by either.

**Consequences.**

- `ReviewReport.findings` deviates from `docs/spec.md` §7.3's list-typed shape **permanently**. Spec.md §7.3 carries the same `Amended 2026-05-28 — see DECISIONS.md#030` inline-comment shape that #028 introduced for TriageResult's fifth field — but compact (pointer + field shape only); the rationale lives in this entry. Future readers of spec.md who hit the §7.3 block see the amendment pointer and follow the anchor for rationale.

- `ReviewMetrics` deviates from spec.md §7.3's required-int / required-float LLM-aggregate fields as a **V1 transition state**. The amendment pointer explicitly cites FUP-093 as the trigger that reverts the deviation. Distinct from #028 (permanent deviation) and from #016 (permanent expansion) — this entry has a documented exit condition built in.

- `FOLLOWUPS.md#FUP-093`'s **Trigger** paragraph cross-cites `DECISIONS.md#030` as the canonical-record anchor for the V1 nullable shape (the cross-citation lands in the same commit that adds this entry, mirroring the `#027` + `FUP-068` cross-citation pattern). When FUP-093 closes, the spec.md §7.3 amendment pointers for the four LLM-aggregate fields delete and the field annotations revert — this is a closeout artifact tied to the FUP.

- Dashboard + replay are unaffected by the nullable deviation. Both read audit truth from `LLMCallEvent` rows joined on `review_id`; the denormalized `ReviewMetrics` snapshot is a convenience, not a source of truth. Cost-budget alerting (when it lands) must read audit truth too, not ReviewMetrics — V1 enforcement is in the module docstring at `src/outrider/schemas/review_report.py`.

- The tuple-not-list pattern is now uniformly applied across all four cross-boundary models carrying finding-like collections (`PRContext.changed_files`, `HITLDecision.decisions`, `AnalysisRound.findings`, `ReviewReport.findings`). Any future schema gaining a similar collection (V1.5 `OpenAIProvider` audit-aggregate models, V2 cross-PR-pattern-memory containers) inherits the precedent.

- The pattern this entry establishes — "V1 transition-state with documented exit condition tied to a FUP" — is reusable. Future deviations that exist only because a dependent module hasn't yet landed can follow the same amendment shape (cite the FUP, name the trigger, declare the revert path) rather than encoding the transition as a permanent canonical change.

**Referenced from.** `src/outrider/schemas/review_report.py` (the `findings: tuple[...]` field + the four nullable LLM-aggregate fields + the canonical-shape note in the module docstring), `src/outrider/schemas/analysis_round.py` (sibling tuple-findings precedent), `src/outrider/schemas/pr_context.py` (sibling tuple-collection precedent), `src/outrider/schemas/hitl.py` (sibling tuple-collection precedent, per `#014`), `src/outrider/agent/nodes/synthesize.py` (the `_compute_metrics` call site that populates the nullable aggregates as `None` in V1; the FUP-093 trigger site for revert), `FOLLOWUPS.md#FUP-093` (the transition-state exit condition for the nullable LLM-aggregate deviation; Trigger paragraph cross-cites this decision).
