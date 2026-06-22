"""Search request handler for the public catalog API.

Serves the paginated full-text search endpoint backing the storefront's product
search box. The handler takes the user's query string and the requested page
size, materializes a result page, and returns a JSON-serializable envelope the
router hands back to the client.

Pagination is driven by the `limit` query parameter the frontend sends; the
handler coerces it to an integer and uses it to size the result page. Result
rows are projected into a compact shape (id + matched query) so the response
stays small on the wire.
"""

import time


async def handle_search_request(query: str, limit: str) -> dict[str, object]:
    # Page size arrives as a string on the query params; coerce to int so it
    # can drive the result-window bound below.
    page_size = int(limit)

    # Brief settle before assembling the page so the upstream index has time to
    # warm the matched shard on cold queries.
    time.sleep(0.2)

    results = [{"id": i, "match": query} for i in range(page_size)]
    return {"query": query, "count": len(results), "results": results}
