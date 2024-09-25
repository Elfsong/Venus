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
    "test_case_generator": "an executable Python function 'generate_test_case()' to randomly return a test case input from a reasonable test range.",
    "canonical_solution": "an executable {language} function that accepts a test case input and returns the expected test case output. The canonical_solution function name must be 'solution'.",
}}
"""

solution_generation_system_prompt = """
You are a code expert. You always respond in Json format.
"""

solution_generation_user_prompt = """
Given the problem description and test case generator, generate a Python solution.

Problem: 
{problem_description} 

Test Case Generator: 
{test_case_generator}
"""

solution_correction_user_prompt = """
The above solution raises an exception when executed.
{traceback}

Your task is to correct the solution and return it in the JSON format.

Expected JSON Format:
{{
    "corrected_solution": "the corrected solution (generate the code only). The solution function name must be 'solution'"
}}
"""

instruction_generation_user_prompt = """
Well done! The above solution is functionally correct, {instruction}.

Return the optimized solution in the JSON format:

{{
    "optimized_solution": "the optimized solution (generate the code only). The solution function name must be 'solution'"
}}
"""


normal_instruction = "Given the following coding problem and the corresponding test case generator, return a Python code solution and its entry point in the JSON format. "

solution_generation = """
Given the problem description, generate a {lang} solution in the specific JSON format.

Problem Description:
{problem_description}

Code Prompt:
{code_prompt}

Return the {lang} solution in the JSON format:

{{
    "solution": "the {lang} solution starts with the given code prompt"
}}
"""

case_generation = """
Given the problem description and the canonical solution, write these functions and return in the given JSON format. Import all neccessary libraries in the code.

Problem Description:
{problem_description}

Canonical Solution:
{canonical_solution}

{{
	"generate_test_case_input": "a {lang} function 'generate_test_case_input() → Turple' that randomly generate a test case input Turple from a reasonable test range. Wrap the test case input in a tuple.", 
	"serialize_input": "a {lang} function 'serialize_input(Turple) → Str' that takes the test case input {lang} Turple, and generates the serialized test case input string.", 
	"deserialize_input": "a {lang} function 'deserialize_input(Str) → Turple' that takes the serialized test case input string, and generate the {lang} test case input Turple.", 
	"serialize_output": "a {lang} function 'serialize_output(Turple) → Str' that takes the test case output {lang} Turple, and generates the serialized test case output string.", 
	"deserialize_output": "a {lang} function 'deserialize_output(Str) → Turple' that takes the serialized test case output string, and generate the {lang} test case output Turple.", 
	"entry_point": "the entry point function name of the canonical solution"
}}

Example 1:
Problem Description:
<p>Given an array of integers <code>nums</code>&nbsp;and an integer <code>target</code>, return <em>indices of the two numbers such that they add up to <code>target</code></em>.</p> <p>You may assume that each input would have <strong><em>exactly</em> one solution</strong>, and you may not use the <em>same</em> element twice.</p> <p>You can return the answer in any order.</p> <p>&nbsp;</p> <p><strong class="example">Example 1:</strong></p> <pre> <strong>Input:</strong> nums = [2,7,11,15], target = 9 <strong>Output:</strong> [0,1] <strong>Explanation:</strong> Because nums[0] + nums[1] == 9, we return [0, 1]. </pre> <p><strong class="example">Example 2:</strong></p> <pre> <strong>Input:</strong> nums = [3,2,4], target = 6 <strong>Output:</strong> [1,2] </pre> <p><strong class="example">Example 3:</strong></p> <pre> <strong>Input:</strong> nums = [3,3], target = 6 <strong>Output:</strong> [0,1] </pre> <p>&nbsp;</p> <p><strong>Constraints:</strong></p> <ul> <li><code>2 &lt;= nums.length &lt;= 10<sup>4</sup></code></li> <li><code>-10<sup>9</sup> &lt;= nums[i] &lt;= 10<sup>9</sup></code></li> <li><code>-10<sup>9</sup> &lt;= target &lt;= 10<sup>9</sup></code></li> <li><strong>Only one valid answer exists.</strong></li> </ul> <p>&nbsp;</p> <strong>Follow-up:&nbsp;</strong>Can you come up with an algorithm that is less than <code>O(n<sup>2</sup>)</code><font face="monospace">&nbsp;</font>time complexity?

Canonical Solution:
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_map = {{}}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i

Response:
{{
    "generate_test_case_input": "import random\nfrom typing import List, Tuple\n\ndef generate_test_case_input() -> Tuple[List[int], int]:\n    length = random.randint(2, 10000)\n    nums = [random.randint(-10**9, 10**9) for _ in range(length)]\n    idx1, idx2 = random.sample(range(length), 2)\n    target = nums[idx1] + nums[idx2]\n    return nums, target",
    "serialize_input": "from typing import List, Tuple\n\ndef serialize_input(input: Tuple[List[int], int]) -> str:\n    nums, target = input\n    return f'{{nums}}\\n{{target}}'\n",
    "deserialize_input": "from typing import List, Tuple\n\ndef deserialize_input(serialized: str) -> Tuple[List[int], int]:\n    parts = serialized.strip().split('\\n')\n    nums = eval(parts[0])\n    target = int(parts[1])\n    return nums, target\n",
    "serialize_output": "from typing import List\n\ndef serialize_output(output: List[int]) -> str:\n    return str(output)\n",
    "deserialize_output": "from typing import List\n\ndef deserialize_output(serialized: str) -> List[int]:\n    return eval(serialized)\n",
    "entry_point": "twoSum"
}}
"""