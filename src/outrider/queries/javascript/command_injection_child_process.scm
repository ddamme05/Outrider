; Shell command built from dynamic strings at an exec sink. Two patterns:
; the bare form (`exec`/`execSync` from a destructured require/import) and
; the namespace member form (`child_process.exec` / `cp.execSync`). Both
; require the first argument to be a `+` concatenation or a template literal
; WITH a substitution — a constant command string is not injectable, and a
; bare identifier argument on the bare form would over-claim unrelated `exec`
; functions.
;
; Anchor-capture protocol (BindingRule mode="anchor_import"): the member arm
; captures its receiver as @_recv; the producer admits a match only when
; @_recv (else the bare @_fn) is bound by an import from the query's module
; set. The import join replaces the previous literal `child_process`
; receiver check, so aliased namespaces (`const cp = require('child_process');
; cp.exec(...)`) are now provable instead of a recall gap. Forms with no
; provable binding do not match (JUDGED covers each; FUP-214): nested /
; inline-require / parenthesized / `this.`-qualified / TS non-null
; receivers, member-chain require (`const exec =
; require('child_process').exec`) and dynamic `await import()` (neither
; yields an ImportRef), and aliased NAMED imports
; (`import { exec as run }` — the literal name anchor never sees `run`).
(call_expression
  function: (identifier) @_fn
  arguments: (arguments
    .
    [
      (binary_expression operator: "+")
      (template_string (template_substitution))
    ] @_arg)
  (#any-of? @_fn "exec" "execSync")) @command_injection_child_process

(call_expression
  function: (member_expression
    object: (identifier) @_recv
    property: (property_identifier) @_fn)
  arguments: (arguments
    .
    [
      (binary_expression operator: "+")
      (template_string (template_substitution))
    ] @_arg)
  (#any-of? @_fn "exec" "execSync")) @command_injection_child_process
