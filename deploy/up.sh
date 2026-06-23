#!/usr/bin/env bash
# Launch the keyless demo with fail-loud preflight checks. Run from the droplet:
#   cd /opt/outrider/deploy && bash up.sh
#
# Guards the three footguns a first live deploy hits, each of which produces an
# obscure failure instead of a clear one:
#   1. A missing/empty deploy/demo_seed.sql. The compose bind-mounts it into
#      /docker-entrypoint-initdb.d; if absent, Docker silently creates a
#      root-owned DIRECTORY there and Postgres aborts init ("Is a directory"),
#      crash-looping behind restart: unless-stopped — a 100%-dead demo.
#   2. An unedited DEMO_DOMAIN placeholder. Caddy would loop on ACME failures for
#      a domain you don't control and can burn the Let's Encrypt rate limit.
#   3. A missing .env or a placeholder admin token (the app rejects change-me at
#      startup; catching it here beats a crash-loop).
set -euo pipefail
cd "$(dirname "$0")"

fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

if [ -d demo_seed.sql ]; then
  fail "deploy/demo_seed.sql is a DIRECTORY — a previous launch with a missing seed created it. Run:
    docker compose -f docker-compose.demo.yml down -v && sudo rm -rf demo_seed.sql
  then scp the real seed (scripts/demo_fixtures/demo_seed.sql) into place and re-run."
fi
[ -f demo_seed.sql ] || fail "deploy/demo_seed.sql is missing. From your laptop:
    scp scripts/demo_fixtures/demo_seed.sql root@<droplet-ip>:/opt/outrider/deploy/demo_seed.sql"
[ -s demo_seed.sql ] || fail "deploy/demo_seed.sql is empty — re-scp the real ~795K dump."
[ -f .env ] || fail "deploy/.env is missing. Create it (see .env.demo.example)."

domain=$(grep -E '^DEMO_DOMAIN=' .env | tail -1 | cut -d= -f2-)
token=$(grep -E '^OUTRIDER_ADMIN_API_KEY=' .env | tail -1 | cut -d= -f2-)
[ -n "$domain" ] || fail "DEMO_DOMAIN is unset in .env."
[ "$domain" != "demo.example.com" ] || fail "DEMO_DOMAIN is still the placeholder demo.example.com — set your real domain in .env."
[ "$token" != "change-me" ] || fail "OUTRIDER_ADMIN_API_KEY is still the placeholder change-me — set a real token in .env."

echo "Preflight OK — seed $(wc -c < demo_seed.sql) bytes, DEMO_DOMAIN=$domain. Launching..."
exec docker compose -f docker-compose.demo.yml up -d --build
