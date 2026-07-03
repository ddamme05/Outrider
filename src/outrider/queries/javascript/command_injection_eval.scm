; Dynamic code execution: `eval(<non-literal>)` and `new Function(...)` with
; at least one non-literal argument. The argument alternation enumerates the
; dynamic-expression node types (mirroring the Python eval/exec query): a
; string literal or substitution-free template argument is constant code and
; does not fire. Member `eval` (`math.eval(...)`) is excluded — third-party
; evaluators are a JUDGED call, not the global. `new Function` with all-literal
; arguments is likewise constant-code and excluded.
(call_expression
  function: (identifier) @_fn
  arguments: (arguments
    .
    [
      (identifier)
      (member_expression)
      (subscript_expression)
      (call_expression)
      (binary_expression)
      (template_string (template_substitution))
      (await_expression)
      (ternary_expression)
      (parenthesized_expression)
    ] @_arg)
  (#eq? @_fn "eval")) @command_injection_eval

(new_expression
  constructor: (identifier) @_ctor
  arguments: (arguments
    [
      (identifier)
      (member_expression)
      (subscript_expression)
      (call_expression)
      (binary_expression)
      (template_string (template_substitution))
      (await_expression)
      (ternary_expression)
      (parenthesized_expression)
    ] @_arg)
  (#eq? @_ctor "Function")) @command_injection_eval
