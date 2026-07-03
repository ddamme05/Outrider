; Weak hash construction (`crypto.createHash("md5"|"sha1")`) anchored to the
; call USE. The function alternation matches both the bare form (`createHash`,
; from a destructured require/import) and the member form (`crypto.createHash`),
; so the algorithm predicate lives in one place. Case-insensitive: Node routes
; the name through OpenSSL's case-insensitive lookup, so "MD5" is the same
; construction. Non-literal algorithm arguments are a deliberate recall gap
; (JUDGED covers them) — a name-based OBSERVED claim needs the literal.
;
; Anchor-capture protocol (BindingRule mode="anchor_import"): the member arm
; captures its receiver as @_recv; the producer admits a match only when
; @_recv (else the bare @_fn) is bound by an import from the query's module
; set. The receiver is constrained to a simple identifier — a nested
; receiver (`a.b.createHash`) has no provable binding, so it does not match
; (JUDGED covers it).
(call_expression
  function: [
    (identifier) @_fn
    (member_expression
      object: (identifier) @_recv
      property: (property_identifier) @_fn)
  ]
  arguments: (arguments
    .
    (string (string_fragment) @_algo))
  (#eq? @_fn "createHash")
  (#match? @_algo "(?i)^(md5|sha1)$")) @weak_crypto_hash
