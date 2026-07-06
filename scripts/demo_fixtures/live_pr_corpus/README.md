# Catalog Service

The catalog service is the read-heavy backend behind the storefront's product
browsing and search experience. It fronts the merchandising database, serves the
paginated full-text search endpoint the storefront's search box calls, and
assembles the product-detail payloads the web and mobile clients render. Every
request that touches a shopper — a category listing, a search page, a product
detail view — is served from here, so the service is tuned for low tail latency
and horizontal scale rather than write throughput.

## Overview

Products, variants, pricing, and availability are owned by upstream inventory and
merchandising systems and land here through a change-data-capture stream. The
catalog service treats that stream as its source of truth: it materializes a
denormalized read model optimized for the exact shapes the storefront asks for,
keeps a search index in sync, and never writes back to the systems of record.
This one-way flow keeps the read path simple and lets us rebuild the read model
from the stream whenever the projection logic changes.

The public surface is a small, versioned HTTP API. Clients send a search query
and pagination hints; the service returns a compact JSON envelope of matched
product summaries plus the cursors needed to page through results. Detail
endpoints hydrate a single product into the richer shape the product page needs —
media, variant matrix, breadcrumb trail, and merchandising badges.

## Architecture

The service is organized around three layers:

- **API layer** — request handlers that validate incoming query parameters,
  translate them into internal query objects, and project results into the
  wire shapes the clients expect. This is the only layer that speaks HTTP.
- **Query layer** — a thin data-access seam over the read model and the search
  index. It owns the SQL and the index queries and returns domain rows; the API
  layer never assembles queries itself.
- **Projection layer** — the consumer that reads the change-data-capture stream
  and keeps the read model and search index current.

Keeping these seams distinct means the storefront's contract can evolve at the
API layer without reshaping the read model, and the read model can be reindexed
without touching request handling.

## Getting started

The service runs as a standard Python application backed by Postgres for the read
model and an external search index for full-text queries. Local development
brings both up through the project's compose file, applies the schema
migrations, and starts the API with autoreload:

```bash
docker compose up -d postgres search-index
alembic upgrade head
uvicorn catalog.main:app --reload
```

With the stack up, the API is available on `http://localhost:8000`. The search
endpoint accepts a `query` string and a `limit` page size; the detail endpoint
takes a product id. Interactive API docs are served at `/docs`.

## Configuration

Runtime configuration is read from the environment. The values most worth knowing
about on a fresh checkout:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Read-model Postgres connection string. |
| `SEARCH_INDEX_URL` | Endpoint for the full-text search index. |
| `MAX_PAGE_SIZE` | Upper bound the API clamps requested page sizes to. |
| `STREAM_CONSUMER_GROUP` | Consumer group the projection layer joins. |

Defaults suitable for local development ship in `.env.example`; copy it to `.env`
and adjust the connection strings if you are not using the bundled compose stack.

## Running the checks

The suite is split into fast in-process checks and integration checks that need
the compose stack running:

```bash
pytest catalog/checks/unit
pytest catalog/checks/integration
```

The unit tier needs no services and runs in a few seconds. The integration tier
expects Postgres and the search index to be reachable, and exercises the query
layer and the projection consumer end to end against real backends.

## Deployment

The service ships as a single container image and scales out behind the storefront
edge. Because it is read-only with respect to the systems of record, instances are
interchangeable and can be added or replaced without coordination — a rolling
deploy is safe by construction. The projection consumer runs as a separate
long-lived process so that indexing lag never backs up request handling.
