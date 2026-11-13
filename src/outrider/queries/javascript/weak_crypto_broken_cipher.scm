; Match createCipheriv/createCipher with a literal broken-cipher algorithm name.
;
; Recognize DES/3DES, RC2, RC4, and Blowfish prefixes, case-insensitively,
; ending at a hyphen or the end of the name. The producer must bind @_recv or
; @_fn to an allowed import. Nested receivers and nonliteral algorithm names
; are not covered and need model review.
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
