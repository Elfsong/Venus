# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/06

problem_synthesis_system_prompt = """
You are a code expert. You can create hard coding problems.
"""

problem_synthesis_user_prompt = """
Using the following code snippets as a reference, create an expert-level coding problem, its corresponding test case generators and a canonical solution. Wrap the test case input in a tuple.

For each 'input = generate_test_case()', the corresponding output must be 'output = solution(*input)'.

Code snippets: {code_reference}

Return in the JSON format:
{{
    "problem_description": "a problem description with input/output examples and constraints in Markdown.",
    "simple_test_case_generator": "an executable Python function 'generate_test_case()' to randomly return a test case input from a reasonable test range",
    "full_test_case_generator": "an executable Python function 'generate_test_case()' to randomly return a test case input from the full input range",
    "canonical_solution": "an executable {language} function that accepts a test case input and returns the expected test case output. The solution function name must be 'solution'.",
}}
"""

solution_generation_system_prompt = """
You are a code expert. Don't use markdown in the response.
"""

solution_generation_user_prompt = """
{instruction} 

{problem_description}

Return in the JSON format:
{{
    "canonical_solution": "a pure {language} function that accepts a test case input and returns the expected test case output.",
    "entry_point": "the entry point of the canonical_solution."
}}
"""

normal_instruction = "Given the following coding problem and the corresponding test case generator, return a Python code solution and its entry point in the JSON format. "