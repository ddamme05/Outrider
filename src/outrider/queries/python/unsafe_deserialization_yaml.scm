(call
  function: (attribute
    object: (identifier) @_obj
    attribute: (identifier) @_meth)
  arguments: (argument_list) @_args
  (#eq? @_obj "yaml")
  (#eq? @_meth "load")
  (#not-match? @_args "Loader")) @unsafe_deserialization_yaml
