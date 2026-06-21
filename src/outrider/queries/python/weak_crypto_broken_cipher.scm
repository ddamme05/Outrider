(call
  function: (attribute
    object: (identifier) @_cipher
    attribute: (identifier) @_method)
  (#any-of? @_cipher "DES" "ARC4" "RC4" "Blowfish")
  (#eq? @_method "new")) @weak_crypto_broken_cipher
