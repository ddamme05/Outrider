; Match SQL concatenation or template substitution in .query/.execute calls.
;
; Concatenation needs a string operand directly beside +; templates need a
; substitution. Bare query helpers and .exec are excluded: .exec can be a
; shell-command sink. Other SQL call forms still need model review.
; The producer requires an allowed database-driver import in the file.
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
