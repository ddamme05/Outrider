; TLS certificate verification disabled. Three patterns: the option pair
; `rejectUnauthorized: false` (identifier or string key — both object-literal
; spellings), and the process-wide kill switch
; `NODE_TLS_REJECT_UNAUTHORIZED = "0"` in both the dot and bracket
; environment forms. Only the literal `false` / literal `"0"` fire — a
; variable value is a JUDGED contextual call.
(pair
  key: [
    (property_identifier) @_key
    (string (string_fragment) @_key)
  ]
  value: (false)
  (#eq? @_key "rejectUnauthorized")) @tls_verify_disabled

(assignment_expression
  left: (member_expression
    property: (property_identifier) @_env)
  right: (string (string_fragment) @_val)
  (#eq? @_env "NODE_TLS_REJECT_UNAUTHORIZED")
  (#eq? @_val "0")) @tls_verify_disabled

(assignment_expression
  left: (subscript_expression
    index: (string (string_fragment) @_env))
  right: (string (string_fragment) @_val)
  (#eq? @_env "NODE_TLS_REJECT_UNAUTHORIZED")
  (#eq? @_val "0")) @tls_verify_disabled
