(call
  function: (attribute
    attribute: (identifier) @_ctor)
  arguments: (argument_list
    (attribute
      attribute: (identifier) @_mode))
  (#eq? @_ctor "new")
  (#eq? @_mode "MODE_ECB")) @weak_crypto_ecb_mode

(call
  function: (identifier) @_ctor2
  arguments: (argument_list
    (call
      function: (attribute
        attribute: (identifier) @_m)))
  (#eq? @_ctor2 "Cipher")
  (#eq? @_m "ECB")) @weak_crypto_ecb_mode
