; Match concatenated or interpolated arguments to exec/execSync.
;
; The producer must bind @_recv (member calls) or @_fn (bare calls) to a
; child_process import. Simple namespace aliases work. Nested or inline
; receivers and renamed function imports are not covered and need model review
; (FUP-214). Plain literal command strings do not match.
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
