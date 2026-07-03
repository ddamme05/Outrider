; Weak hash construction (`crypto.createHash("md5"|"sha1")`) anchored to the
; call USE. The function alternation matches both the bare form (`createHash`,
; from a destructured require/import) and the member form (`crypto.createHash`),
; so the algorithm predicate lives in one place. Case-insensitive: Node routes
; the name through OpenSSL's case-insensitive lookup, so "MD5" is the same
; construction. Non-literal algorithm arguments are a deliberate recall gap
; (JUDGED covers them) — a name-based OBSERVED claim needs the literal.
(call_expression
  function: [
    (identifier) @_fn
    (member_expression
      property: (property_identifier) @_fn)
  ]
  arguments: (arguments
    .
    (string (string_fragment) @_algo))
  (#eq? @_fn "createHash")
  (#match? @_algo "(?i)^(md5|sha1)$")) @weak_crypto_hash
