; Cipher constructed in ECB mode: `createCipheriv`/`createCipher` with a
; literal algorithm name ending in `-ecb` (`aes-128-ecb`, `des-ecb`, ...).
; Suffix-anchored so `-ecb` mid-name does not fire. A `des-ecb` literal fires
; this AND the broken-cipher query — same FindingType + line, so the
; producer's content-hash collapse keeps one finding. Encryption-construction
; sites only (matching the broken-cipher anchor set).
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
  (#match? @_algo "(?i)-ecb$")) @weak_crypto_ecb_mode
