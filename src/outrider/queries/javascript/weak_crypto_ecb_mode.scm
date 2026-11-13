; Match createCipheriv/createCipher with a literal algorithm ending in -ecb.
;
; The suffix match is case-insensitive and intentionally overlaps broken-cipher
; matches such as des-ecb. The producer must bind @_recv or @_fn to an allowed
; import; nested receivers and nonliteral names still need model review.
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
  (#match? @_algo "(?i)-ecb$")) @weak_crypto_ecb_mode
