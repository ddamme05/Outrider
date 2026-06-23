# Deploy the keyless public demo to a DigitalOcean droplet

The demo box runs Outrider in **DEMO_MODE** — a keyless, read-only dashboard over the
seeded reviews. It holds **no** Anthropic / GitHub / Slack credentials and runs no
reviews. Three containers on one droplet:

- **postgres** — the demo database, auto-restored from `demo_seed.sql` on first boot.
- **app** — the FastAPI app in DEMO_MODE (only `OUTRIDER_ADMIN_API_KEY` + `DATABASE_URL`).
- **web** — Caddy: serves the dashboard SPA, proxies `/api`, terminates HTTPS.

## 1 · Provision the droplet
- DigitalOcean → Create Droplet → **Ubuntu 24.04**, smallest plan (~$6/mo — the demo
  is light, no LLM calls). Add your SSH key.
- Point a DNS **A record** (e.g. `demo.yourdomain.com`) at the droplet IP. (Skip if
  you'll serve plain HTTP on the bare IP.)

## 2 · Install Docker (on the droplet)
```bash
ssh root@<droplet-ip>
curl -fsSL https://get.docker.com | sh
```

## 3 · Copy the repo + the seed
From your laptop. Use `git archive` so **only tracked files** reach the public box — a
broad `rsync` would also copy local artifacts (`.venv`, `.claude`, `.codex`, `.agents`,
`AUDIT_LOG.md`, the gitignored `docs/`, stray `.env` files), which don't belong on a
public demo host:
```bash
ssh root@<droplet-ip> 'mkdir -p /opt/outrider'
git -C ~/projects/outrider archive --format=tar HEAD \
  | ssh root@<droplet-ip> 'tar -x -C /opt/outrider'
# the seed is gitignored (so not in the archive) — copy it into deploy/ separately:
scp ~/projects/outrider/scripts/demo_fixtures/demo_seed.sql \
  root@<droplet-ip>:/opt/outrider/deploy/demo_seed.sql
```

## 4 · Configure + launch
```bash
ssh root@<droplet-ip>
cd /opt/outrider/deploy
cp .env.demo.example .env
nano .env     # set POSTGRES_PASSWORD, OUTRIDER_ADMIN_API_KEY, DEMO_DOMAIN
bash up.sh    # preflight-checks the seed + .env, then `docker compose up -d --build`
```
First boot builds the app + dashboard images and Postgres auto-restores the seed.
`up.sh` fails loud if `deploy/demo_seed.sql` is missing — a silent miss makes Docker
create a root-owned directory there and crash-loops Postgres. Caddy fetches a Let's
Encrypt certificate for `DEMO_DOMAIN` automatically.

## 5 · Firewall
```bash
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
```

## 6 · Verify
- `https://demo.yourdomain.com/health` → `{"status":"ok"}`. Caddy starts before the app
  finishes booting, so the first hit may return **502 for ~15s** — retry, it's not a failure.
- Open `https://demo.yourdomain.com`, paste `OUTRIDER_ADMIN_API_KEY` when prompted →
  the 6 seeded reviews appear. Three park at `awaiting_approval` (the HITL gate); the rest
  publish. The 28-file breadth review carries the broad taxonomy; run a **replay** on any
  review to watch it reconstruct.

## Re-seeding
The seed restores only onto a fresh data volume. Drop **only** the DB volume — `down -v`
would also wipe `caddy-data` and force a Let's Encrypt re-issue (rate-limit risk):
```bash
docker compose -f docker-compose.demo.yml stop postgres
docker compose -f docker-compose.demo.yml rm -f postgres
docker volume rm outrider-demo_demo-data
# replace deploy/demo_seed.sql, then:
bash up.sh
```

## Notes
- **Keyless by design.** No LLM/GitHub/Slack secrets exist on this box. The only
  credential is `OUTRIDER_ADMIN_API_KEY`, a read-gate over public seed data.
- **Frictionless viewing.** Share a one-click link with the token in the URL fragment —
  `https://outrider-review.duckdns.org/#token=<OUTRIDER_ADMIN_API_KEY>`. The dashboard
  adopts the token and strips the fragment on load (supplied at runtime via the link,
  never baked into the bundle per `DECISIONS.md#011`). Or just print the token next to
  the link and let viewers paste it into the gate.
- **HTTP-only / no domain.** Set `DEMO_DOMAIN=:80` to serve plain HTTP on the IP (no
  cert) — fine for a quick demo, but prefer a domain + HTTPS for anything public.
- **Updating the demo data** is just a re-seed (above) — the box never needs the
  review pipeline or any credentials to refresh.
