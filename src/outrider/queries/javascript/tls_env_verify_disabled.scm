; Process-wide TLS verification kill switch:
; `process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"` in the dot and bracket
; forms. The receiver is constrained to the `process.env` global in the
; query itself, so the pattern is self-proving and carries no BindingRule —
; the kill switch needs no import (split out of `tls_verify_disabled` so
; that query's module_presence gate cannot wrongly suppress this one).
; Only the literal `"0"` fires — a variable value is a JUDGED contextual
; call. Deliberate recall gaps (JUDGED covers each): mixed forms
; (`process["env"].NODE_TLS_...`), aliased receivers (`const env =
; process.env; env.NODE_TLS_... = "0"`), and `globalThis.process.env` —
; the receiver is constrained to the literal `process.env` chain.
;
; Known residual (FUP-214): the producer's scope-containment gate admits
; matches only inside an extracted ScopeUnit (function/method/class), and
; the CANONICAL real-world form of this kill switch is a module-top-level
; statement — which has no enclosing scope unit and is dropped there. The
; producer test suite pins that veto explicitly; the fix shape is a
; changed-region admission arm for self-proving module-level queries.
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
