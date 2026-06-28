; Broken/legacy cipher construction (DES/3DES/RC2/RC4/Blowfish) anchored to the
; `.new(...)` USE, not the import. The object alternation
; `[(identifier) (attribute attribute: (identifier))]` matches both the bare form
; (`DES.new`) and the qualified-import form (`Crypto.Cipher.DES.new`), so the cipher
; #any-of? list lives in one place (no bare/qualified duplication to drift).
; CAST/IDEA are deliberately EXCLUDED — common English words, too collision-prone
; for a name-based OBSERVED finding; the JUDGED path covers them in context.
(call
  function: (attribute
    object: [(identifier) @_cipher (attribute attribute: (identifier) @_cipher)]
    attribute: (identifier) @_method)
  (#any-of? @_cipher "DES" "DES3" "ARC4" "RC4" "ARC2" "RC2" "Blowfish")
  (#eq? @_method "new")) @weak_crypto_broken_cipher
