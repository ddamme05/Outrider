; Match rejectUnauthorized: false with an identifier or string key.
;
; The producer requires a TLS-capable module in the file before admitting this
; option pair. Variable values need model review. The process-wide environment
; setting has its own query, tls_env_verify_disabled.scm, with different guards.
(pair
  key: [
    (property_identifier) @_key
    (string (string_fragment) @_key)
  ]
  value: (false)
  (#eq? @_key "rejectUnauthorized")) @tls_verify_disabled
