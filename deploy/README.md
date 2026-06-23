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
From your laptop:
```bash
rsync -a --exclude .git --exclude 'dashboard/node_modules' --exclude '.env' \
  ~/projects/outrider/ root@<droplet-ip>:/opt/outrider/
# the seed is gitignored — copy it into deploy/ where compose expects it:
scp ~/projects/outrider/scripts/demo_fixtures/demo_seed.sql \
  root@<droplet-ip>:/opt/outrider/deploy/demo_seed.sql
```

## 4 · Configure + launch
```bash
ssh root@<droplet-ip>
cd /opt/outrider/deploy
cp .env.demo.example .env
nano .env     # set POSTGRES_PASSWORD, OUTRIDER_ADMIN_API_KEY, DEMO_DOMAIN
docker compose -f docker-compose.demo.yml up -d --build
```
First boot builds the app + dashboard images and Postgres auto-restores the seed.
Caddy fetches a Let's Encrypt certificate for `DEMO_DOMAIN` automatically.

## 5 · Firewall
```bash
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
```

## 6 · Verify
- `https://demo.yourdomain.com/health` → `{"status":"ok"}`
- Open `https://demo.yourdomain.com`, paste `OUTRIDER_ADMIN_API_KEY` when prompted →
  the 5 seeded reviews appear. Two park at `awaiting_approval` (the HITL gate), three
  publish; run a **replay** on any review to watch it reconstruct.

## Re-seeding
The seed restores only onto a fresh data volume:
```bash
docker compose -f docker-compose.demo.yml down -v      # drops the demo data volume
# replace deploy/demo_seed.sql, then:
docker compose -f docker-compose.demo.yml up -d --build
```

## Notes
- **Keyless by design.** No LLM/GitHub/Slack secrets exist on this box. The only
  credential is `OUTRIDER_ADMIN_API_KEY`, a read-gate over public seed data.
- **Frictionless viewing (optional).** To skip the token prompt in a recorded demo,
  share the token on your landing page, or bake a default into the dashboard build.
- **HTTP-only / no domain.** Set `DEMO_DOMAIN=:80` to serve plain HTTP on the IP (no
  cert) — fine for a quick demo, but prefer a domain + HTTPS for anything public.
- **Updating the demo data** is just a re-seed (above) — the box never needs the
  review pipeline or any credentials to refresh.
