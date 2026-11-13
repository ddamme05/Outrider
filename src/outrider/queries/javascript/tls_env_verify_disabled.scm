; Match process.env.NODE_TLS_REJECT_UNAUTHORIZED set to the literal string "0".
;
; Support dot access and a bracketed final key without requiring an import.
; The producer rejects shadowed process bindings, including value imports;
; re-exports and type-only imports do not create a runtime binding. Its file-level
; import check can also suppress a sibling use after a function-local rebind
; (FUP-214; DECISIONS.md#060).
;
; Module-level matches must be outside parsed scopes and wholly within added
; head lines. Unchanged lines do not qualify (DECISIONS.md#062). Aliased, mixed
; bracket/dot, and globalThis receivers are not covered and need model review.
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
