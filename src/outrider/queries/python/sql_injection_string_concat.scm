(call
  function: (attribute
    attribute: (identifier) @_method)
  arguments: (argument_list
    .
    [
      (string (interpolation))
      (binary_operator left: (string))
      (call
        function: (attribute
          object: (string)
          attribute: (identifier) @_fmt)
        arguments: (argument_list (_))
        (#eq? @_fmt "format"))
    ] @_sql)
  (#any-of? @_method "execute" "executemany")) @sql_injection_string_concat
