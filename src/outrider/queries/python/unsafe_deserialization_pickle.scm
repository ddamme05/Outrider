(call
  function: (attribute
    object: (identifier) @_mod
    attribute: (identifier) @_fn)
  (#any-of? @_mod "pickle" "cPickle")
  (#any-of? @_fn "load" "loads")) @unsafe_deserialization_pickle
