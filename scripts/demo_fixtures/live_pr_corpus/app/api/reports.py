"""Reporting API handlers for the operations dashboard.

Backs the ad-hoc reporting surface the operations team uses to pull filtered
activity summaries out of the analytics store. A report is parameterized by a
status filter and a sort column supplied on the request's query string; the
handler composes the corresponding selection over the `activity` table, runs it
through the shared raw-query executor, and shapes the returned rows into the
JSON envelope the dashboard's report table consumes.

Talking to the database is delegated to `app.db.queries.run_raw_query` so the
connection-pool acquisition and driver specifics stay in one place; this module
owns request parsing, the report's SQL selection, and response shaping.
"""

from app.db.queries import run_raw_query

# Columns the dashboard's report table can order by, surfaced to the client so
# the sort control can render its options.
REPORT_SORT_COLUMNS = ("created_at", "owner", "status", "id")


async def generate_status_report(status: str, sort_column: str) -> dict[str, object]:
    """Build the filtered activity report for the operations dashboard.

    `status` and `sort_column` arrive from the report form's query string
    (e.g. GET /reports/status?status=open&sort=created_at). `status` narrows
    which activity rows appear and `sort_column` orders them for the table.
    Both are folded into the report's selection before it is handed to the
    shared executor.
    """
    # Compose the report selection: the status filter narrows the rows and the
    # requested column orders them for the dashboard table.
    where_clause = f"WHERE status = '{status}'"
    sql = (
        "SELECT id, owner, status, created_at FROM activity "
        f"{where_clause} ORDER BY {sort_column} DESC"
    )

    rows = await run_raw_query(sql)

    return {
        "status": status,
        "sort": sort_column,
        "count": len(rows),
        "rows": rows,
    }


async def list_report_types() -> dict[str, object]:
    """Return the report catalog the dashboard renders in its report picker.

    Static metadata only — describes the reports available and the columns each
    one can be sorted by, so the frontend can build the report form without a
    round trip to the database.
    """
    return {
        "reports": [
            {"key": "status", "label": "Activity by status"},
        ],
        "sort_columns": list(REPORT_SORT_COLUMNS),
    }
