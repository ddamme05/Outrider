import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

const advisory = {
  url: "https://github.com/advisories/GHSA-fc5q-39xc-vjpf",
  name: "picomatch",
  dependency: "picomatch",
  severity: "high",
};

const packages = {
  "node_modules/picomatch": { version: "4.0.7", dev: true },
  "node_modules/tinyglobby": { version: "0.2.17", dev: true },
  "node_modules/vite": { version: "8.1.7", dev: true },
};

const findings = {
  picomatch: { name: "picomatch", nodes: ["node_modules/picomatch"], via: [advisory] },
  tinyglobby: { name: "tinyglobby", nodes: ["node_modules/tinyglobby"], via: ["picomatch"] },
  vite: { name: "vite", nodes: ["node_modules/vite"], via: ["tinyglobby", "picomatch"] },
};

function report(vulnerabilities) {
  return {
    auditReportVersion: 2,
    vulnerabilities,
    metadata: { vulnerabilities: { total: Object.keys(vulnerabilities).length } },
  };
}

function runAudit(t, auditReport, {
  lockPackages = packages,
  now = "2026-12-01T23:59:59Z",
  npmStatus = 1,
} = {}) {
  const root = mkdtempSync(join(tmpdir(), "outrider-dashboard-audit-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const directory of ["scripts", "dashboard", "bin"]) {
    mkdirSync(join(root, directory));
  }

  const checker = join(root, "scripts", "check_dashboard_audit.mjs");
  copyFileSync(new URL("./check_dashboard_audit.mjs", import.meta.url), checker);
  writeFileSync(join(root, "dashboard", "package-lock.json"), JSON.stringify({ packages: lockPackages }));
  writeFileSync(
    join(root, "bin", "npm"),
    '#!/bin/sh\nprintf "%s" "$AUDIT_TEST_REPORT"\nexit "$AUDIT_TEST_STATUS"\n',
    { mode: 0o755 },
  );

  const clock = join(root, "clock.mjs");
  writeFileSync(clock, "Date.now = () => Date.parse(process.env.AUDIT_TEST_NOW);\n");
  const result = spawnSync(process.execPath, ["--import", clock, checker], {
    cwd: root,
    encoding: "utf8",
    env: {
      PATH: join(root, "bin"),
      AUDIT_TEST_REPORT: JSON.stringify(auditReport),
      AUDIT_TEST_STATUS: String(npmStatus),
      AUDIT_TEST_NOW: now,
    },
    timeout: 10_000,
  });
  assert.ifError(result.error);
  return result;
}

test("accepts the scoped dev-only advisory and its dependent findings before expiry", (t) => {
  const result = runAudit(t, report(findings));
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Accepted GHSA-fc5q-39xc-vjpf/);
});

test("rejects the advisory at its expiry time", (t) => {
  const result = runAudit(t, report(findings), { now: "2026-12-02T00:00:00Z" });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /exception expired/);
});

test("rejects a separate advisory alongside the waived dependency chain", (t) => {
  const additional = {
    ...findings,
    vite: {
      ...findings.vite,
      via: [
        ...findings.vite.via,
        { ...advisory, name: "vite", dependency: "vite", url: "https://example.com/advisory" },
      ],
    },
  };
  const result = runAudit(t, report(additional));
  assert.equal(result.status, 1);
  assert.match(result.stderr, /outside the temporary picomatch exception/);
});

test("rejects the advisory when an affected dependent is in the production graph", (t) => {
  const lockPackages = { ...packages, "node_modules/vite": { version: "8.1.7" } };
  const result = runAudit(t, report(findings), { lockPackages });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /outside the temporary picomatch exception/);
});

test("rejects a registry error even when npm uses the advisory exit code", (t) => {
  const result = runAudit(t, { error: { code: "E503", summary: "registry unavailable" } });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Expected a complete npm audit report/);
});

test("accepts a clean report after the exception expires", (t) => {
  const result = runAudit(t, report({}), { now: "2026-12-02T00:00:00Z", npmStatus: 0 });
  assert.equal(result.status, 0, result.stderr);
  assert.doesNotMatch(result.stdout, /Accepted GHSA/);
});
