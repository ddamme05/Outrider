(function_definition
  "async"
  body: (block (expression_statement [
    (call
      function: [
        (attribute object: (identifier) @_t (#eq? @_t "time")
                   attribute: (identifier) @_ta (#eq? @_ta "sleep"))
        (attribute object: (identifier) @_r (#eq? @_r "requests")
                   attribute: (identifier) @_ra
                   (#any-of? @_ra "get" "post" "put" "delete" "head" "patch" "request"))
        (identifier) @_fn (#eq? @_fn "open")
      ]) @blocking_call_in_async
    (assignment right:
      (call
        function: [
          (attribute object: (identifier) @_t2 (#eq? @_t2 "time")
                     attribute: (identifier) @_ta2 (#eq? @_ta2 "sleep"))
          (attribute object: (identifier) @_r2 (#eq? @_r2 "requests")
                     attribute: (identifier) @_ra2
                     (#any-of? @_ra2 "get" "post" "put" "delete" "head" "patch" "request"))
          (identifier) @_fn2 (#eq? @_fn2 "open")
        ]) @blocking_call_in_async)
  ])))
