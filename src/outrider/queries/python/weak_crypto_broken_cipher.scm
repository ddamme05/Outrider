(call
  function: (attribute
    object: (identifier) @_cipher
    attribute: (identifier) @_method)
  (#any-of? @_cipher "DES" "DES3" "ARC4" "RC4" "ARC2" "RC2" "Blowfish" "CAST" "IDEA")
  (#eq? @_method "new")) @weak_crypto_broken_cipher

(call
  function: (attribute
    object: (attribute
      attribute: (identifier) @_cipher)
    attribute: (identifier) @_method)
  (#any-of? @_cipher "DES" "DES3" "ARC4" "RC4" "ARC2" "RC2" "Blowfish" "CAST" "IDEA")
  (#eq? @_method "new")) @weak_crypto_broken_cipher
