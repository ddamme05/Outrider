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
; Every pattern pins the call's LIBRARY object to RSA/DSA so the finding's
; claim ("an RSA or DSA key < 2048 bits") is structurally proven — `generate`
; and `generate_private_key`/`bits=`/`key_size=` are NOT RSA/DSA-exclusive
; names on their own (a benign `token.generate(bits=128)` must not over-claim).

; --- PyCryptodome: RSA.generate(...) / DSA.generate(...), capitalized object ---

; Positional: RSA.generate(1024). The key size is the first positional arg
; (signature: generate(bits, ...)), so the `.` anchor pins @_keysize to the
; first argument — the `e=` exponent (also an integer) can never be the size.
(call
  function: (attribute
    object: (identifier) @_lib
    attribute: (identifier) @_method)
  arguments: (argument_list
    .
    (integer) @_keysize)
  (#any-of? @_lib "RSA" "DSA")
  (#eq? @_method "generate")) @weak_asymmetric_key_size

; Positional, qualified object: Crypto.PublicKey.RSA.generate(1024)
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

; Keyword: RSA.generate(bits=1024)
(call
  function: (attribute
    object: (identifier) @_libk
    attribute: (identifier) @_methodk)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kwk
      value: (integer) @_keysize))
  (#any-of? @_libk "RSA" "DSA")
  (#eq? @_methodk "generate")
  (#eq? @_kwk "bits")) @weak_asymmetric_key_size

; Keyword, qualified object: Crypto.PublicKey.RSA.generate(bits=1024)
(call
  function: (attribute
    object: (attribute
      attribute: (identifier) @_libkq)
    attribute: (identifier) @_methodkq)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kwkq
      value: (integer) @_keysize))
  (#any-of? @_libkq "RSA" "DSA")
  (#eq? @_methodkq "generate")
  (#eq? @_kwkq "bits")) @weak_asymmetric_key_size

; --- cryptography: rsa.generate_private_key(key_size=...), lowercase module ---
; (ec.generate_private_key takes a curve, not key_size; dh uses
; generate_parameters — so the rsa/dsa object pin + key_size keyword are exact.)

; Keyword: rsa.generate_private_key(public_exponent=65537, key_size=1024)
(call
  function: (attribute
    object: (identifier) @_libc
    attribute: (identifier) @_methodc)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kwc
      value: (integer) @_keysize))
  (#any-of? @_libc "rsa" "dsa")
  (#eq? @_methodc "generate_private_key")
  (#eq? @_kwc "key_size")) @weak_asymmetric_key_size

; Keyword, qualified object:
;   cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key(key_size=1024)
(call
  function: (attribute
    object: (attribute
      attribute: (identifier) @_libcq)
    attribute: (identifier) @_methodcq)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kwcq
      value: (integer) @_keysize))
  (#any-of? @_libcq "rsa" "dsa")
  (#eq? @_methodcq "generate_private_key")
  (#eq? @_kwcq "key_size")) @weak_asymmetric_key_size
