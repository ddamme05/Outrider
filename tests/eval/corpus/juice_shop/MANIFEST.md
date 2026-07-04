# Juice Shop graded corpus — provenance

Vendored subset of OWASP Juice Shop, the ground-truth corpus for grading the JS/TS
OBSERVED catalog (`specs/2026-07-04-juice-shop-graded-corpus.md`).

## Upstream

- **Repository:** https://github.com/juice-shop/juice-shop
- **Commit:** `33518f5a0911e25d9df747b1e70fb7af279a755c` (branch `master`, fetched 2026-07-04)
- **License:** MIT. The upstream `LICENSE` file is vendored verbatim alongside this
  manifest at `tests/eval/corpus/juice_shop/LICENSE` (the authoritative copy — the
  header note below is a pointer, not a paraphrase).

## Vendored files (verbatim, no content minimization)

Each file is copied byte-for-byte from the upstream path shown; only the directory root
changes (`<upstream>/` → `tests/eval/corpus/juice_shop/src/`). No lines were trimmed —
whole files are kept so grading has real surrounding context and the line numbers in
`ground_truth.json` match upstream.

| Corpus path | Upstream path | Role |
|-------------|---------------|------|
| `src/lib/insecurity.ts` | `lib/insecurity.ts` | weak_crypto TP (`crypto.createHash('md5')`, L41); `createHmac('sha256')` L42 is a natural true-negative |
| `src/routes/captcha.ts` | `routes/captcha.ts` | command_injection eval TP (`eval(expression)`, L22) |
| `src/routes/userProfile.ts` | `routes/userProfile.ts` | command_injection eval TP (`eval(code)`, L61) |
| `src/routes/search.ts` | `routes/search.ts` | SQL injection (`models.sequelize.query(\`…${criteria}…\`)`, L23) — raw-matches, denied at admission (module_presence residual) |
| `src/routes/login.ts` | `routes/login.ts` | SQL injection (`models.sequelize.query(\`…${email}…\`)`, L34) — same residual |
| `src/lib/xml.ts` | `lib/xml.ts` | true-negative: `new Function(<literals>)` L12 (all-literal → not dynamic code) |
| `src/routes/continueCode.ts` | `routes/continueCode.ts` | FP-stress clean: direct `sequelize` import (`Op`), ORM only, no string-concat sink |
| `src/lib/codingChallenges.ts` | `lib/codingChallenges.ts` | FP-stress clean: `new RegExp(…).exec(…)`, no `child_process` binding |

## No network at test time

The files above are checked in. The grading harness (`tests/eval/corpus_grading.py`) reads
them from disk; nothing clones or fetches at test time. Re-vendoring (bumping the pinned
commit) is a manual, reviewed step: re-copy the listed paths, re-run the harness, and
regenerate `tests/eval/scorecard_juice_shop.json` in the same commit.

## LICENSE

The upstream MIT license is vendored verbatim at
`tests/eval/corpus/juice_shop/LICENSE` (copied byte-for-byte from the pinned commit's
repo-root `LICENSE`). That file is authoritative; this manifest does not paraphrase it.
