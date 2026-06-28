; Weak RSA/DSA asymmetric key size (< 2048 bits). The .scm matches the
; key-generation call shape and captures the integer-LITERAL key size as
; @_keysize; the registry value-predicate (queries/value_predicates.py) then
; confirms the literal is below the secure threshold. A non-literal size
; (a variable / expression) never matches — the pattern requires an (integer)
; node — so the OBSERVED proof stays deterministic and never over-claims on a
; size it cannot evaluate.

; PyCryptodome positional: RSA.generate(1024) / DSA.generate(1024).
; The key size is the first positional arg (signature: generate(bits, ...)),
; so the `.` anchor pins @_keysize to the first argument — the `e=` exponent
; (also an integer) can never be mistaken for the key size.
(call
  function: (attribute
    object: (identifier) @_lib
    attribute: (identifier) @_method)
  arguments: (argument_list
    .
    (integer) @_keysize)
  (#any-of? @_lib "RSA" "DSA")
  (#eq? @_method "generate")) @weak_asymmetric_key_size

; PyCryptodome positional, qualified object: Crypto.PublicKey.RSA.generate(1024)
(call
  function: (attribute
    object: (attribute
      attribute: (identifier) @_libq)
    attribute: (identifier) @_methodq)
  arguments: (argument_list
    .
    (integer) @_keysize)
  (#any-of? @_libq "RSA" "DSA")
  (#eq? @_methodq "generate")) @weak_asymmetric_key_size

; PyCryptodome keyword: RSA.generate(bits=1024) (object bare or qualified)
(call
  function: (attribute
    attribute: (identifier) @_methodk)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kwk
      value: (integer) @_keysize))
  (#eq? @_methodk "generate")
  (#eq? @_kwk "bits")) @weak_asymmetric_key_size

; cryptography: rsa.generate_private_key(..., key_size=1024) /
; dsa.generate_private_key(..., key_size=1024). The key_size keyword pins it
; to RSA/DSA (ec.generate_private_key takes a curve, not a key_size).
(call
  function: (attribute
    attribute: (identifier) @_methodc)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kwc
      value: (integer) @_keysize))
  (#eq? @_methodc "generate_private_key")
  (#eq? @_kwc "key_size")) @weak_asymmetric_key_size
