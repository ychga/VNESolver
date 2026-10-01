1. Parse the `userIntent` to determine:
   - Intent type (`intentType`).
   - Appropriate action (`Action`) to execute.
   - Necessary parameters (`Params`) for the action.
2. Ensure the output matches the format: a list of tuples `(intentType, Action, Params)`.
3. Only defined user intentType and Action can be used
4. If the action involves GML generation, ensure the output adheres to GML format.
5. Handle vague or incomplete intents by making reasonable assumptions based on historical examples and provided constraints.
6. Ensure that the generated gml file meets the given constraints and reflects the user’s intent.