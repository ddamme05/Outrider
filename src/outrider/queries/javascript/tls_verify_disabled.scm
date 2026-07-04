; TLS certificate verification disabled: the option pair
; `rejectUnauthorized: false` (identifier or string key — both object-literal
; spellings). Only the literal `false` fires — a variable value is a JUDGED
; contextual call. The process-wide NODE_TLS_REJECT_UNAUTHORIZED kill switch
; lives in its own query (`javascript.tls_env_verify_disabled`): its
; `process.env` receiver is text-constrained in the query, needs no
; import, and is shadow-guarded by the producer (see that query's
; header), while this
; pair pattern is gated by the producer on file-level presence of a module
; that honors the option (BindingRule mode="module_presence") — an option
; object with no TLS-capable consumer in the file is a JUDGED call.
(pair
  key: [
    (property_identifier) @_key
    (string (string_fragment) @_key)
  ]
  value: (false)
  (#eq? @_key "rejectUnauthorized")) @tls_verify_disabled
