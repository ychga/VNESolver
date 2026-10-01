1. Ensure the output adheres to GML format.
2. Return only the plain text gml content, no other content. Make sure it starts with 'graph [' and ends with ']'
3. Do not return data in markdown format, which will cause it to be wrapped in '''
4. Handle vague or incomplete intents by making reasonable assumptions based on historical examples and provided constraints.
5. Ensure that the resources allocated in generated gml file meets the given constraints and reflects the user’s intent.