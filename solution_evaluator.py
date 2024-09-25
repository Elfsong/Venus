# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-09-25

import random
from src.sandbox import Sandbox
from datasets import load_dataset


class SolutionEvaluator:
    def __init__(self):
        self.sandbox = Sandbox()
        self.ds = load_dataset("Elfsong/venus_case", "python3")
        
    def evaluate(self, solution, test_cases, functions):               
        sample = {
            "timeout": 120, 
            "solution": solution,
            "functions": functions,
            "test_cases": test_cases,
        }
        results = self.sandbox.run_code_execution(sample)
        if results[0] == 'pass' and len(results) == 3:
            return results
        else:
            return results[0], None, None

    def evaluate_pipeline(self):
        for p_index, instance in enumerate(self.ds['train']):
            print("================ {question_id}.{question_name} [{index}/{total}]".format(question_id=instance['question_id'], question_name=instance['name'], index=p_index+1, total=len(self.ds['train'])))
            
            solutions = instance['rt_list'] + instance['mm_list']
            solution_candidates = list()
            for solution in solutions:
                code = solution['code']
                if 'stdin' not in code and 'Solution' in code:
                    solution_candidates += [code]
            solution_candidates = random.sample(solution_candidates, 5)

            for s_index, solution in enumerate(solution_candidates):
                status, rt, mm = self.evaluate(solution, instance['test_cases'], instance['test_case_functions'])
                if status == 'pass':
                    print(f"[{s_index+1}/{len(solution_candidates)}] Passed 🟢 [{str(round(rt, 4))} ms]\t[{str(round(mm, 2))} kb]")
                else:
                    print(f"[{s_index+1}/{len(solution_candidates)}] Failed 🔴[{status}]")
    
if __name__ == "__main__":
    evaluator = SolutionEvaluator()
    evaluator.evaluate_pipeline()