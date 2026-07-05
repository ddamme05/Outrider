; Process-wide TLS verification kill switch:
; `process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"` in the dot and bracket
; forms. The receiver is constrained to the `process.env` identifier chain
; in the query itself, so the pattern carries no BindingRule — the kill
; switch needs no import (split out of `tls_verify_disabled` so that
; query's module_presence gate cannot wrongly suppress this one). The
; text constraint is completed by the registry entry's
; `shadow_guard=("process",)`: the producer denies a match whose `process`
; is locally rebound — a parameter or module-scope declaration whose
; visibility span contains the match, OR a module-scope VALUE
; import/require rebind (`const process = require("./mock")`) — a
; re-export or `import type` of the name creates no runtime binding
; and does not deny. Residual
; (safe-direction, FUP-214): a FUNCTION-scope require-rebind of `process`
; over-denies a sibling use (file-level import check), degrading to JUDGED.
; See DECISIONS.md#060.
; Only the literal `"0"` fires — a variable value is a JUDGED contextual
; call. Deliberate recall gaps (JUDGED covers each): mixed forms
; (`process["env"].NODE_TLS_...`), aliased receivers (`const env =
; process.env; env.NODE_TLS_... = "0"`), and `globalThis.process.env` —
; the receiver is constrained to the literal `process.env` chain.
;
; Module-top-level reach (DECISIONS.md#062):
; the CANONICAL real-world form of this kill switch — a module-top-level
; statement in index.js/config — is admitted by the producer's module-scope
; arm (this query is `module_scope_eligible`): a match disjoint from every
; parsed scope and fully inside a head-side added-line range produces the
; OBSERVED finding; a diff whose added lines are all module-level routes
; to the degraded review
; instead of skipping. Matches on UNCHANGED module-level lines stay
; excluded — the diff anchors the proof.
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
