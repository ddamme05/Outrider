import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

const dashboard = new URL("../dashboard/", import.meta.url);
const advisory = "https://github.com/advisories/GHSA-fc5q-39xc-vjpf";
const expiration = Date.parse("2026-12-02T00:00:00Z");

function matchesException(findings, packages) {
  const picomatch = findings.get("picomatch");
  const cause = picomatch?.via?.[0];
  if (
    picomatch?.via?.length !== 1 ||
    cause?.url !== advisory ||
    cause.name !== "picomatch" ||
    cause.dependency !== "picomatch" ||
    cause.severity !== "high"
  ) {
    return false;
  }

  for (const [name, finding] of findings) {
    const path = "node_modules/" + name;
    if (
      finding?.name !== name ||
      !Array.isArray(finding.via) ||
      finding.via.length === 0 ||
      !Array.isArray(finding.nodes) ||
      finding.nodes.length === 0 ||
      !finding.nodes.every((node) => (
        typeof node === "string" &&
        (node === path || node.endsWith("/" + path)) &&
        Object.hasOwn(packages, node) &&
        packages[node]?.dev === true &&
        (name !== "picomatch" || packages[node].version === "4.0.7")
      ))
    ) {
      return false;
    }
  }

  // npm reports impacted dependents as findings that refer to another package.
  // Work outward from picomatch so missing links, cycles and other advisories fail.
  const covered = new Set(["picomatch"]);
  while (covered.size < findings.size) {
    const previousSize = covered.size;
    for (const [name, finding] of findings) {
      if (finding.via.every((dependency) => covered.has(dependency))) {
        covered.add(name);
      }
    }
    if (covered.size === previousSize) {
      return false;
    }
  }
  return true;
}

function checkAudit() {
  const result = spawnSync(
    "npm",
    ["audit", "--package-lock-only", "--ignore-scripts", "--json", "--audit-level=info", "--include=dev"],
    { cwd: dashboard, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] },
  );
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0 && result.status !== 1) {
    throw new Error("npm audit did not complete.");
  }

  const report = JSON.parse(result.stdout);
  if (
    report?.auditReportVersion !== 2 ||
    Object.hasOwn(report, "error") ||
    !report.vulnerabilities ||
    typeof report.vulnerabilities !== "object" ||
    Array.isArray(report.vulnerabilities)
  ) {
    throw new Error("Expected a complete npm audit report.");
  }

  const findings = new Map(Object.entries(report.vulnerabilities));
  console.log(JSON.stringify(report, null, 2));
  if (findings.size !== report.metadata?.vulnerabilities?.total) {
    throw new Error("npm audit reported an inconsistent finding count.");
  }
  if (result.status === 0 && findings.size === 0) {
    return;
  }
  if (result.status !== 1 || findings.size === 0) {
    throw new Error("npm audit exit status did not match its findings.");
  }
  if (Date.now() >= expiration) {
    throw new Error("The temporary picomatch advisory exception expired.");
  }

  if (existsSync(new URL("npm-shrinkwrap.json", dashboard))) {
    throw new Error("The temporary exception applies only to dashboard/package-lock.json.");
  }

  const { packages } = JSON.parse(readFileSync(new URL("package-lock.json", dashboard), "utf8"));
  if (!packages || !matchesException(findings, packages)) {
    throw new Error("npm audit found issues outside the temporary picomatch exception.");
  }

  // Snyk revoked SNYK-JS-PICOMATCH-21248802. The production image
  // copies the compiled dashboard, and this waiver covers only locked dev tools.
  console.log("Accepted GHSA-fc5q-39xc-vjpf for dev-only picomatch@4.0.7 until 2026-12-02 UTC.");
}

try {
  checkAudit();
} catch (error) {
  console.error("Dashboard dependency audit failed:", error.message);
  process.exitCode = 1;
}
