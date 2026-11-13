; Match the listed expression forms in global eval calls and Function constructors.
;
; Plain string arguments and templates without substitutions do not match.
; Member calls such as math.eval are excluded because they may use a
; third-party evaluator rather than execute JavaScript.
; The producer rejects locally shadowed eval or Function bindings.
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
