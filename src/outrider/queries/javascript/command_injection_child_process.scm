; Shell command built from dynamic strings at an exec sink. Two patterns:
; the bare form (`exec`/`execSync` from a destructured require/import) and the
; require-namespace member form (`child_process.exec`/`.execSync`). Both
; require the first argument to be a `+` concatenation or a template literal
; WITH a substitution — a constant command string is not injectable, and a
; bare identifier argument on the bare form would over-claim unrelated `exec`
; functions. Aliased namespaces (`cp.exec`) are a deliberate recall gap: they
; need symbol resolution, not a name pattern (JUDGED covers them).
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
    object: (identifier) @_mod
    property: (property_identifier) @_fn)
  arguments: (arguments
    .
    [
      (binary_expression operator: "+")
      (template_string (template_substitution))
    ] @_arg)
  (#eq? @_mod "child_process")
  (#any-of? @_fn "exec" "execSync")) @command_injection_child_process
