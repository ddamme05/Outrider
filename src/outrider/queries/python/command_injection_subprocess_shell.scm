(call
  function: (attribute
    attribute: (identifier) @_method
    (#any-of? @_method "run" "call" "Popen" "check_output" "check_call"))
  arguments: (argument_list
    (keyword_argument
      name: (identifier) @_kw
      value: (true)
      (#eq? @_kw "shell")))) @command_injection_subprocess_shell
