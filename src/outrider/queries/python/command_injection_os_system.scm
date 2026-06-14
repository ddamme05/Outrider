(call
  function: (attribute
    object: (identifier) @_obj
    attribute: (identifier) @_attr)
  (#any-of? @_obj "os" "posix")
  (#any-of? @_attr "system" "popen")) @command_injection_os_system
