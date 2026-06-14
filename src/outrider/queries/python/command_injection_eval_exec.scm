(call
  function: (identifier) @_fn
  arguments: (argument_list
    .
    [
      (identifier)
      (attribute)
      (call)
      (subscript)
      (binary_operator)
      (boolean_operator)
      (comparison_operator)
      (unary_operator)
      (not_operator)
      (parenthesized_expression)
      (conditional_expression)
      (await)
      (list)
      (dictionary)
      (set)
      (tuple)
      (list_comprehension)
      (dictionary_comprehension)
      (set_comprehension)
      (lambda)
      (named_expression)
      (string (interpolation))
    ] @_arg)
  (#any-of? @_fn "eval" "exec")) @command_injection_eval_exec
