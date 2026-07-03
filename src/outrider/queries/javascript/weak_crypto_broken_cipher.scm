; Broken/legacy cipher construction (DES / 3DES / RC2 / RC4 / Blowfish)
; anchored to the `createCipheriv` / legacy `createCipher` USE with a literal
; OpenSSL algorithm name. The regex anchors the family prefix at a `-` or
; end-of-string boundary, so `des-ede3-cbc` fires while `desx-cbc` does not;
; case-insensitive to mirror OpenSSL's name lookup. Non-literal algorithm
; names are a deliberate recall gap (JUDGED covers them).
;
; Anchor-capture protocol (BindingRule mode="anchor_import"): the member arm
; captures its receiver as @_recv; the producer admits a match only when
; @_recv (else the bare @_fn) is bound by an import from the query's module
; set. Nested receivers (`a.b.createCipheriv`) have no provable binding and
; do not match (JUDGED covers them).
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
  (#any-of? @_fn "createCipheriv" "createCipher")
  (#match? @_algo "(?i)^(des|rc2|rc4|bf|blowfish)(-|$)")) @weak_crypto_broken_cipher
