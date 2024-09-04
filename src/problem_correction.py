# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/04

from datasets import Dataset, load_dataset

class Code_Corrector():
    def __init__(self) -> None:
        self.datasets = load_dataset()
        self.refine_loop_count = 2
    
    def solution_refine(self):
        pass
    
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
                "content": f"Given the problem and the test case generator, generate a Python Solution.\nProblem: {problem_description}\nTest Case Generator: {test_case_generator}" 
            }]
            
            interactions += [{
                "role": "coder",
                "content": canonical_solution
            }]
            
            while status != "success" and self.refine_loop_count > 0:
                interactions += [{
                    "role": "user",
                    "content": f'Error {traceback} when running the solution. Try to fix it.'
                }]
                result = self.solution_refine()
                canonical_solution = result["canonical_solution"]
                interactions += [{
                    "role": "coder",
                    "content": canonical_solution
                }]
                
                
                
                
            
            
            

                