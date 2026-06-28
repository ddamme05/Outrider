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

(call
  function: (attribute
    attribute: (identifier) @_ctor3)
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kw3
      value: (attribute
        attribute: (identifier) @_mode3)))
  (#eq? @_ctor3 "new")
  (#eq? @_kw3 "mode")
  (#eq? @_mode3 "MODE_ECB")) @weak_crypto_ecb_mode

(call
  function: (identifier) @_ctor4
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kw4
      value: (call
        function: (attribute
          attribute: (identifier) @_m4))))
  (#eq? @_ctor4 "Cipher")
  (#eq? @_kw4 "mode")
  (#eq? @_m4 "ECB")) @weak_crypto_ecb_mode
