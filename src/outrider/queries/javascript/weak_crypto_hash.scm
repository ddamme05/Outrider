; Match createHash with a literal md5 or sha1 algorithm name, ignoring case.
;
; The producer must bind @_recv or @_fn to an allowed crypto import. Simple
; namespace aliases work; renamed function imports, nested or inline receivers,
; and nonliteral algorithm names are not covered and need model review
; (FUP-214).
(call_expression
  function: [
    (identifier) @_fn
    (member_expression
      object: (identifier) @_recv
      property: (property_identifier) @_fn)
  ]
  arguments: (arguments
    .
    (string (string_fragment) @_algo))
  (#eq? @_fn "createHash")
  (#match? @_algo "(?i)^(md5|sha1)$")) @weak_crypto_hash
