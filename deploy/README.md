# Deploy the keyless public demo to a DigitalOcean droplet

The demo server runs Outrider in DEMO_MODE, with a read-only dashboard showing
saved example reviews. It holds no Anthropic / GitHub / Slack credentials and runs
no reviews. Three containers run on one DigitalOcean server (a droplet):

- **postgres** restores the demo database from `demo_seed.sql` on first boot.
- **app** runs FastAPI in DEMO_MODE (only `OUTRIDER_ADMIN_API_KEY` + `DATABASE_URL`).
- **web** runs Caddy, which serves the dashboard, forwards `/api` requests to the
  app, and provides HTTPS.

## 1 · Provision the droplet

- DigitalOcean → Create Droplet → Ubuntu 24.04, smallest plan (~$6/mo; the demo
  makes no model calls). Add your SSH key.
- Point a DNS A record (e.g. `demo.yourdomain.com`) at the droplet IP. Skip this if
  you'll serve plain HTTP on the IP.

## 2 · Install Docker (on the droplet)

```bash
ssh root@<droplet-ip>
curl -fsSL https://get.docker.com | sh
```

## 3 · Copy the repo + the seed

From your laptop, use `git archive` so **only tracked files** reach the public
server. Copying the whole directory with `rsync` would also copy local files
(`.venv`, `.claude`, `.codex`, `.agents`, `AUDIT_LOG.md`, the gitignored `docs/`,
stray `.env` files), which don't belong on a public demo server:

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

First boot builds the app and dashboard images, and Postgres restores the saved
reviews. `up.sh` stops with an error if `deploy/demo_seed.sql` is missing. Without
that check, Docker would create a root-owned directory there and Postgres would
keep crashing. For a domain, Caddy fetches a Let's Encrypt certificate for
`DEMO_DOMAIN` automatically.

## 5 · Firewall

```bash
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
```

## 6 · Verify

- `https://demo.yourdomain.com/health` → `{"status":"ok"}`. Caddy starts before the app
  finishes booting, so the first request may return 502 for ~15s. Retry after that.
- Open `https://demo.yourdomain.com`, paste `OUTRIDER_ADMIN_API_KEY` when prompted →
  the 6 saved reviews appear. Two wait at `awaiting_approval` for a human decision.
  One already has a decision recorded when the demo data is created, including a
  severity change and the later publication steps. The others show recorded
  publication attempts.
  The 28-file review includes a wide range of finding types. Run a replay on any
  review to watch its recorded steps.

## Re-seeding

The saved reviews restore only onto a fresh database volume. Remove **only** the
database volume. `down -v` would also delete `caddy-data` and force Let's Encrypt
to issue a new certificate, which could hit its rate limit:

```bash
docker compose -f docker-compose.demo.yml stop postgres
docker compose -f docker-compose.demo.yml rm -f postgres
docker volume rm outrider-demo_demo-data
# replace deploy/demo_seed.sql, then:
bash up.sh
```

## Notes

- No model/GitHub/Slack secrets exist on this server. `OUTRIDER_ADMIN_API_KEY`
  controls access to viewing the public demo data.
- Share a one-click link with the token after the # in the URL:
  `https://outrider-review.duckdns.org/#token=<OUTRIDER_ADMIN_API_KEY>`. The dashboard
  reads the token and removes that part of the URL on load. The link supplies the
  token when the page opens; it is never built into the dashboard files, per
  `DECISIONS.md#011`. Or print the token next to the link so viewers can paste it
  into the access prompt.
- Set `DEMO_DOMAIN=:80` to serve plain HTTP on the IP without a certificate.
  This works for a quick demo; prefer a domain and HTTPS for anything public.
- To update the demo data, repeat the re-seeding steps above. The server needs
  no review runs or model/GitHub/Slack credentials to load the replacement data.
