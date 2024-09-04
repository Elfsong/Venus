# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/04

from datasets import Dataset, load_dataset

class Code_Corrector():
    def __init__(self) -> None:
        self.datasets = load_dataset()
    
    
    def pipeline(self) -> None:
        for instance in self.datasets:
            interactions = list()
            
            problem_description = instance["problem_description"]
            test_case_generator = instance["simple_test_case_generator"]
            canonical_solution = instance["canonical_solution"]
            status = instance["status"]
            traceback = instance["traceback"]
            
            interactions += [{
                "role": "user",
                "content": f"Given the problem and the test case generator, generate a Python Solution.\nProblem: {problem_description}\ntest" 
            }]
            

                