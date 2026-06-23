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
  fail "deploy/demo_seed.sql is a DIRECTORY — a previous launch with a missing seed created it. Run
  (drops ONLY the bad DB volume — keeps the Caddy cert, matching the re-seed path):
    docker compose -f docker-compose.demo.yml down
    docker volume rm outrider-demo_demo-data
    sudo rm -rf demo_seed.sql
  then scp the real seed (scripts/demo_fixtures/demo_seed.sql) into place and re-run."
fi
[ -f demo_seed.sql ] || fail "deploy/demo_seed.sql is missing. From your laptop:
    scp scripts/demo_fixtures/demo_seed.sql root@<droplet-ip>:/opt/outrider/deploy/demo_seed.sql"
[ -s demo_seed.sql ] || fail "deploy/demo_seed.sql is empty — re-scp the real ~795K dump."
[ -f .env ] || fail "deploy/.env is missing. Create it (see .env.demo.example)."

# Read KEY=value from .env, stripping ONE layer of surrounding quotes (compose's dotenv
# strips them too, so the preflight sees the same value the app will). The `|| true` keeps
# a grep no-match from aborting the script under `set -e`+`pipefail` before the guards run.
read_env() {
  local v
  v=$(grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true)
  v=${v%\"}; v=${v#\"}; v=${v%\'}; v=${v#\'}
  printf '%s' "$v"
}
domain=$(read_env DEMO_DOMAIN)
token=$(read_env OUTRIDER_ADMIN_API_KEY)

[ -n "$domain" ] || fail "DEMO_DOMAIN is unset in .env."
[ "$domain" != "demo.example.com" ] || fail "DEMO_DOMAIN is still the placeholder demo.example.com — set your real domain in .env."

# Reject the SAME placeholders the app rejects at startup (config.py _PLACEHOLDER_SECRETS),
# case-insensitively, so the preflight isn't narrower than the runtime check it fronts for.
case "$(printf '%s' "$token" | tr '[:upper:]' '[:lower:]')" in
  ""|change-me|changeme|replace-me|replace-me-with-a-long-random-secret|secret|password|your-secret-here)
    fail "OUTRIDER_ADMIN_API_KEY is empty or a known placeholder ('$token') — set a real token in .env (the app rejects these at startup and would crash-loop)." ;;
esac

echo "Preflight OK — seed $(wc -c < demo_seed.sql) bytes, DEMO_DOMAIN=$domain. Launching..."
exec docker compose -f docker-compose.demo.yml up -d --build
