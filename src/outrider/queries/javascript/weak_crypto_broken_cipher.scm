; Broken/legacy cipher construction (DES / 3DES / RC2 / RC4 / Blowfish)
; anchored to the `createCipheriv` / legacy `createCipher` USE with a literal
; OpenSSL algorithm name. The regex anchors the family prefix at a `-` or
; end-of-string boundary, so `des-ede3-cbc` fires while `desx-cbc` does not;
; case-insensitive to mirror OpenSSL's name lookup. Non-literal algorithm
; names are a deliberate recall gap (JUDGED covers them).
(call_expression
  function: [
    (identifier) @_fn
    (member_expression
      property: (property_identifier) @_fn)
  ]
  arguments: (arguments
    .
    (string (string_fragment) @_algo))
  (#any-of? @_fn "createCipheriv" "createCipher")
  (#match? @_algo "(?i)^(des|rc2|rc4|bf|blowfish)(-|$)")) @weak_crypto_broken_cipher
