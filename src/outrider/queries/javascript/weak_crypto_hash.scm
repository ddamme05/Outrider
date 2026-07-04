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
; set. The receiver is constrained to a simple identifier — receivers with
; no provable binding do not match (JUDGED covers each): nested
; (`a.b.createHash`), inline require (`require("crypto").createHash`),
; parenthesized (`(crypto).createHash`), `this.`-qualified, TS non-null
; (`crypto!.createHash`). Aliased NAMED imports
; (`import { createHash as h }`) bind only the alias, which the literal
; name anchor never matches — binding proves receiver/namespace aliases,
; not API-name aliases (FUP-214).
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
