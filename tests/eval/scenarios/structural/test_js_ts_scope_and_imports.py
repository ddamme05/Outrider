"""Structural eval scenario: JS/TS/TSX scope + import extraction.

LLM-free per `docs/conventions.md` — validates `ast_facts` directly.
Required by CLAUDE.md for a new language: realistic JS and TSX sources
must produce well-formed `ScopeUnit`/`ImportRef` objects through the
public `parse_javascript` / `parse_typescript` entry points (per
specs/2026-07-02-js-ts-tree-sitter-adapters.md).
"""

from unittest.mock import MagicMock

from outrider.ast_facts import parse_javascript, parse_typescript

JS_SOURCE = """\
import express from 'express';
import { Router, json as parseJson } from 'express';
import * as path from 'path';
import { validate } from './middleware/validate';
const db = require('../db');

export function createServer(config) {
  const app = express();
  app.use(parseJson());
  return app;
}

export const startServer = async (app, port) => {
  const server = app.listen(port);
  return server;
};

class RequestLogger {
  constructor(sink) { this.sink = sink; }
  log(req) { this.sink.write(format(req)); }
  flush = () => { this.sink.flush(); };
}
"""

# (kind, qualified_name) for every scope the JS fixture must produce —
# exact-set so a walker regression (missed form OR phantom form) fails.
EXPECTED_JS_SCOPES = {
    ("function", "createServer"),
    ("function", "startServer"),
    ("class", "RequestLogger"),
    ("method", "RequestLogger.constructor"),
    ("method", "RequestLogger.log"),
    ("method", "RequestLogger.flush"),
}

# (import_kind, module) per spec Resolution 2's mapping table.
EXPECTED_JS_IMPORTS = {
    ("from", "express"),
    ("direct", "path"),
    ("relative", "./middleware/validate"),
    ("relative", "../db"),
}

TSX_SOURCE = """\
import React from 'react';
import type { AppProps } from './types';

interface CardProps { title: string; body: string; }

export function Card({ title, body }: CardProps): JSX.Element {
  const heading = title.toUpperCase();
  return <section><h2>{heading}</h2><p>{body}</p></section>;
}

export const Page = (props: AppProps) => (
  <main>
    <Card title={props.title} body={props.body} />
  </main>
);
"""

EXPECTED_TSX_SCOPES = {
    ("function", "Card"),
    ("function", "Page"),
}


def test_javascript_scopes_and_imports_extract_structurally() -> None:
    result = parse_javascript(JS_SOURCE.encode(), "src/server.js", MagicMock())
    assert result.parser_outcome == "clean"
    assert result.error_lines == frozenset()
    assert {(s.kind, s.qualified_name) for s in result.scope_units} == EXPECTED_JS_SCOPES
    got_imports = {(i.import_kind, i.module) for i in result.imports}
    # Two ESM statements target 'express' (default + named) — the set
    # collapses them to one ("from", "express") pair; assert the count
    # separately so the double extraction stays visible.
    assert got_imports == EXPECTED_JS_IMPORTS
    assert len(result.imports) == 5
    # Per-kind flag partition (DECISIONS.md#024 Amended 2026-07-03):
    # relative static imports are simple-direct; bare specifiers are not.
    assert all(i.is_simple_direct is (i.import_kind == "relative") for i in result.imports)
    # Calls and assignments attribute to real extracted scopes.
    scope_ids = {s.unit_id for s in result.scope_units}
    assert result.call_sites, "fixture has calls inside scopes"
    assert all(c.enclosing_scope_id in scope_ids for c in result.call_sites)
    assert all(a.enclosing_scope_id in scope_ids for a in result.assignment_sites)


def test_tsx_component_scopes_extract_structurally() -> None:
    result = parse_typescript(TSX_SOURCE.encode(), "src/Page.tsx", MagicMock())
    assert result.parser_outcome == "clean"
    assert result.error_lines == frozenset()
    assert {(s.kind, s.qualified_name) for s in result.scope_units} == EXPECTED_TSX_SCOPES
    got_imports = {(i.import_kind, i.module) for i in result.imports}
    assert got_imports == {("from", "react"), ("relative", "./types")}


def test_flow_typed_js_degrades_to_per_scope_error_signal() -> None:
    """The new-language degradation case: Flow annotations error under
    the JS grammar; the adapter records per-scope has_error +
    error_lines (audited JUDGED fallback is the analyze-dispatch
    follow-up's contract)."""
    source = b"function typed(x: number) { return x; }\nfunction clean(y) { return y; }\n"
    result = parse_javascript(source, "src/flow.js", MagicMock())
    assert result.parser_outcome == "clean"
    by_name = {s.qualified_name: result.has_error[s.unit_id] for s in result.scope_units}
    assert by_name == {"typed": True, "clean": False}
    assert result.error_lines
