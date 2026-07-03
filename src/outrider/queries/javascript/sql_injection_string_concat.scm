; SQL built by concatenation or template substitution at a query call site.
; Anchored to MEMBER calls `.query(...)` / `.execute(...)` (pg/mysql `query`,
; mysql2 `execute`) — the member anchor plus a string operand keeps a bare
; `query(...)` helper or a no-literal `db.query(a + b)` from firing. The
; concat alternation requires a string on either side of `+` (left-assoc
; chains keep a string operand at the outermost node); template literals
; require a substitution. `.exec` is deliberately NOT anchored here: it
; collides with `child_process.exec` (command-injection's sink) and sqlite's
; `.exec` is a JUDGED-covered recall gap.
(call_expression
  function: (member_expression
    property: (property_identifier) @_method)
  arguments: (arguments
    .
    [
      (binary_expression left: (string) operator: "+")
      (binary_expression operator: "+" right: (string))
      (template_string (template_substitution))
    ] @_sql)
  (#any-of? @_method "query" "execute")) @sql_injection_string_concat
