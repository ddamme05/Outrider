; See DECISIONS.md#057 (OBSERVED value-predicate mechanism).
;
; Weak RSA/DSA asymmetric key size (< 2048 bits). The .scm matches the
; key-generation call shape and captures the integer-LITERAL key size as
; @_keysize; the registry value-predicate (queries/value_predicates.py) then
; confirms the literal is below the secure threshold. A non-literal size
; (a variable / expression) never matches — the pattern requires an (integer)
; node — so the OBSERVED proof stays deterministic and never over-claims on a
; size it cannot evaluate.
;
; Every pattern pins the call's LIBRARY object to RSA/DSA so the finding's claim
; ("an RSA or DSA key < 2048 bits") is structurally proven — `generate` /
; `generate_private_key` / `bits=` / `key_size=` are NOT RSA/DSA-exclusive names
; on their own (a benign `token.generate(bits=128)` must not over-claim). The
; object alternation `[(identifier) (attribute attribute: (identifier))]` matches
; both the bare (`RSA`) and qualified (`Crypto.PublicKey.RSA`) object forms in one
; pattern, so the library list lives in a single #any-of? per call shape.
;
; Known recall gap (precision-first; the JUDGED path covers it): the cryptography
; library's POSITIONAL key_size — `rsa.generate_private_key(65537, 1024)` — is not
; matched; only the keyword `key_size=` form is. Closing it needs a second-positional
; anchor (key_size is the 2nd arg, after public_exponent) and is deferred per FUP-193.

; PyCryptodome: RSA.generate(1024) / DSA.generate(1024), capitalized object.
; The key size is the first positional arg (signature: generate(bits, ...)), so the
; `.` anchor pins @_keysize to the first argument — the `e=` exponent (also an
; integer) can never be the size.
(call
  function: (attribute
    object: [(identifier) @_lib (attribute attribute: (identifier) @_lib)]
    attribute: (identifier) @_method)
  arguments: (argument_list
    .
    (integer) @_keysize)
  (#any-of? @_lib "RSA" "DSA")
  (#eq? @_method "generate")) @weak_asymmetric_key_size

; PyCryptodome keyword: RSA.generate(bits=1024)
(call
  function: (attribute
    object: [(identifier) @_lib (attribute attribute: (identifier) @_lib)]
    attribute: (identifier) @_method)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kw
      value: (integer) @_keysize))
  (#any-of? @_lib "RSA" "DSA")
  (#eq? @_method "generate")
  (#eq? @_kw "bits")) @weak_asymmetric_key_size

; cryptography keyword: rsa.generate_private_key(..., key_size=1024), lowercase
; module. (ec.generate_private_key takes a curve, not key_size; dh uses
; generate_parameters — so the rsa/dsa object pin + key_size keyword are exact.)
(call
  function: (attribute
    object: [(identifier) @_lib (attribute attribute: (identifier) @_lib)]
    attribute: (identifier) @_method)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kw
      value: (integer) @_keysize))
  (#any-of? @_lib "rsa" "dsa")
  (#eq? @_method "generate_private_key")
  (#eq? @_kw "key_size")) @weak_asymmetric_key_size
