# Dashboard source walkthrough

Use the dashboard to follow a review, inspect findings, make approval decisions,
and replay recorded work. For the PR-to-GitHub story,
start with [Follow one review in source](../README.md#follow-one-review-in-source).

## Start at the browser entry point

1. [main.tsx](src/main.tsx) reads an optional `#token=` link, sets up data loading,
   and starts the deployment banner, token gate, and page router.
2. [auth/token.ts](src/auth/token.ts) keeps the admin token (the access key) in
   the browser's session storage.
   [TokenGate](src/auth/TokenGate.tsx) asks for a token before showing the pages;
   the backend validates it when requests arrive.
3. [router.tsx](src/router.tsx) maps paths to pages. [App](src/App.tsx) supplies
   the sidebar, top bar, mobile drawer, and space for the selected page.

## Walk through one review

Open `/reviews`, choose a review, and follow one finding through these files.

| Page | What to follow in source |
|---|---|
| `/`: [Overview](src/routes/Overview.tsx) | Load statistics across reviews, then show charts, metric cards, and the approval queue. |
| `/reviews`: [Reviews](src/routes/Reviews.tsx) | Apply shared filters, list reviews, and open one. |
| `/reviews/:reviewId`: [ReviewDetail](src/routes/ReviewDetail.tsx) | Load status, findings, and recorded events; show progress and collect decisions. |
| Finding: [FindingCard](src/components/FindingCard.tsx) | Read evidence, severity, fix suggestions, and where the finding would be posted. |
| Approval: [DecisionControls](src/components/DecisionControls.tsx), [hitl.ts](src/lib/hitl.ts) | Edit and check a decision, then build its request. ReviewDetail submits it; the backend checks it and queues the saved review to resume. |
| `/reviews/:reviewId/replay`: [ReplayReconstruct](src/routes/ReplayReconstruct.tsx) | Play the saved timeline through [ReplayFeed](src/components/ReplayFeed.tsx). Playback makes no new model or GitHub calls. |

ReviewDetail refreshes status and recorded events every two seconds. Findings load
separately; reload the page to fetch updated finding details. The server's saved
approval request lists which findings need a decision. Retrying an accepted
decision sends the same request. [demo.ts](src/lib/demo.ts) requires confirmation
that the server is in production mode before the page enables submission.

## Know where each responsibility lives

- [api/client.ts](src/api/client.ts) loads data and submits changes using the API's
  types. It reads the current token per request and clears it on a 401
  (unauthorized response). Pages choose how often these queries refresh server data.
- [state/filters.ts](src/state/filters.ts) and [state/nav.ts](src/state/nav.ts)
  hold browser display choices. Form drafts and playback state belong to their pages.
- `components/` displays shared page elements; `lib/` holds formatting, finding
  sections, event summaries, and decision rules. [theme.css](src/theme.css) sets styling.
- [SetupGitHubApp](src/routes/SetupGitHubApp.tsx) and
  [ConnectSlack](src/routes/ConnectSlack.tsx) use dedicated clients in `api/`.
  Each client checks responses and handles errors for its setup flow.

## Run and check changes

From `dashboard/`, run `npm run dev` to work on the dashboard, `npm test` for
the existing interaction tests, and `npm run build` to check types and build
production files. Tests use the shared [MSW server](src/test/server.ts) to supply
mock API responses.

For live GitHub/Slack setup, use FastAPI serving the built dashboard with
`OUTRIDER_SERVE_SPA=1`; see the [setup instructions](../README.md#production-setup).
The [Vite proxy](vite.config.ts) covers dashboard API reads, decisions, and privacy;
the backend also needs to receive setup callbacks and GitHub webhooks.

[openapi.json](openapi.json) and [schema.d.ts](src/api/schema.d.ts) are generated.
Regenerate them from backend models with
[gen_openapi.py](../scripts/gen_openapi.py) when the API
changes, following that script's instructions.
