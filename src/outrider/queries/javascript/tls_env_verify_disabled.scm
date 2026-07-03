; Process-wide TLS verification kill switch:
; `process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"` in the dot and bracket
; forms. The receiver is constrained to the `process.env` global in the
; query itself, so the pattern is self-proving and carries no BindingRule —
; the kill switch needs no import (split out of `tls_verify_disabled` so
; that query's module_presence gate cannot wrongly suppress this one).
; Only the literal `"0"` fires — a variable value is a JUDGED contextual
; call. Mixed forms (`process["env"].NODE_TLS_...`) are a deliberate recall
; gap (JUDGED covers them).
(assignment_expression
  left: (member_expression
    object: (member_expression
      object: (identifier) @_proc
      property: (property_identifier) @_envp)
    property: (property_identifier) @_env)
  right: (string (string_fragment) @_val)
  (#eq? @_proc "process")
  (#eq? @_envp "env")
  (#eq? @_env "NODE_TLS_REJECT_UNAUTHORIZED")
  (#eq? @_val "0")) @tls_env_verify_disabled

(assignment_expression
  left: (subscript_expression
    object: (member_expression
      object: (identifier) @_proc
      property: (property_identifier) @_envp)
    index: (string (string_fragment) @_env))
  right: (string (string_fragment) @_val)
  (#eq? @_proc "process")
  (#eq? @_envp "env")
  (#eq? @_env "NODE_TLS_REJECT_UNAUTHORIZED")
  (#eq? @_val "0")) @tls_env_verify_disabled
