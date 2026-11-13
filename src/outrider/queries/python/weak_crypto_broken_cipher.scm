; Match .new calls on DES, DES3, RC2/ARC2, RC4/ARC4, or Blowfish receivers.
;
; Accept bare or qualified receiver names at the call site. CAST and IDEA are
; excluded because these common words make name-only matching too imprecise;
; model review can assess those uses in context.
(call
  function: (attribute
    object: [(identifier) @_cipher (attribute attribute: (identifier) @_cipher)]
    attribute: (identifier) @_method)
  (#any-of? @_cipher "DES" "DES3" "ARC4" "RC4" "ARC2" "RC2" "Blowfish")
  (#eq? @_method "new")) @weak_crypto_broken_cipher
