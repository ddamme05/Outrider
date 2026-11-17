; Candidate only. The source predicate verifies the local import, value,
; and exact condition before the query can produce a finding.
(
  (if_statement
    condition: (boolean_operator operator: "and") @_guard
    consequence: (block
      .
      (expression_statement
        (assignment
          right: (call
            function: (attribute
              object: (call
                function: (attribute attribute: (identifier) @_now))
              attribute: (identifier) @_date)))) @datetime_now_without_offset_check))
  (#eq? @_now "now")
  (#eq? @_date "date")
)
