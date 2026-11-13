; Match RSA/DSA-named key generation with a literal integer key size.
;
; The registry predicate checks whether @_keysize is below 2048 bits
; (DECISIONS.md#057). A variable or expression cannot establish that value.
;
; Support PyCryptodome's first positional size or bits keyword, and
; cryptography's key_size keyword, with bare or qualified receiver names.
; The first-argument anchor keeps an integer exponent from being read as a size.
; cryptography's positional key_size is not covered and needs model review
; (FUP-193).
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
