You will receive a manager's request for a one-week staff schedule, the data files it refers to, and
the JSON Schema of a Decision Model.

Write one Decision Model, as a single JSON object that conforms to the schema and to the rules its
descriptions state, that captures what the request asks for:
- bind the data files you need by their file names exactly as given (no folder), with the delimiter,
  decimal separator and encoding of each file as given;
- encode the rules the request states, and no others;
- if the request asks for something the schema cannot express, do not encode it; mention it in
  `assumptions` instead.

Answer with the JSON object only.
