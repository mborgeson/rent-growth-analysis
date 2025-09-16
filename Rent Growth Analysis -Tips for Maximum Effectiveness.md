**## Usage Tips for Maximum Effectiveness**

\### Prompt Engineering Strategies:

1. Incremental Complexity - Start with bivariate relationships, then expand to multivariate systems
2. Chain-of-Thought for Causality - Ask Claude to reason through economic mechanisms before statistical testing
3. Few-Shot Examples - Include 2-3 example variable pairs with expected relationships

**## Claude-Specific Optimizations:**

\### Include this context setter at prompt start

context = """

You are analyzing economic time series data. Prioritize:

\- Economic theory consistency over pure statistical fit

\- Robust methods over complex models

\- Interpretability for stakeholder communication

"""

**## Common Pitfalls to Avoid:**

1. Don't request all analyses simultaneously - break into phases
2. Specify data format explicitly (wide vs long, index type)
3. Include domain constraints (e.g., "interest rates cannot be negative pre-2008")
