# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/04

import os
import uuid
import sandbox
import pandas as pd
from tqdm import tqdm
from utils import OpenAIClient
from datasets import Dataset, load_dataset

class Code_Corrector():
    def __init__(self) -> None:
        self.datasets = load_dataset("Elfsong/Afterburner", "2e0e3e1a-6ace-11ef-b53a-42010a94000c")["train"]
        self.ds_name = str(uuid.uuid1())
        self.refine_loop_count = 2
        self.model_name = "gpt-4o"
        self.model_token = os.getenv("OPENAI_TOKEN")
        self.sb = sandbox.Sandbox()
        
        self.openai_client = OpenAIClient(model_name=self.model_name, model_token=self.model_token)
    
    def solution_refine(self, messages, traceback):
        prompt = f"""The above solution raises an exception when executed.
            {traceback}
            
            Please correct the solution and return the refined solution in the specified JSON format.

            Expected JSON Format:
            {{
                "refined_solution": "The refined solution that accepts the test case input and returns the expected test case output"
            }}
        """
        
        messages += [
            {"role": "user", "label": "ask_correct", "content": prompt}
        ]
        response = self.openai_client.inference(messages)
        return response
    
    def pipeline(self) -> None:
        data = list()
        
        for instance in tqdm(self.datasets):
            messages = [{
                "role": "system", 
                "label": "system_prompt", 
                "content": "You are a code expert. Generate pure code for code fields."
            }]
            
            problem_description = instance["problem_description"]
            test_case_generator = instance["simple_test_case_generator"]
            canonical_solution = instance["canonical_solution"]
            entry_point = instance["entry_point"]
            status = instance["status"]
            traceback = instance["traceback"]
                        
            messages += [{
                "role": "user",
                "label": "problem",
                "content": f"Given the problem and the test case generator, generate a Python Solution.\nProblem: {problem_description}\nTest Case Generator: {test_case_generator}" 
            }]
            
            messages += [{
                "role": "assistant",
                "label": "init_solution",
                "content": canonical_solution
            }]
            
            refine_loop_count = self.refine_loop_count
            while status != "success" and refine_loop_count > 0:                
                result = self.solution_refine(messages, traceback)
                refined_solution = result["refined_solution"]
                
                messages += [{"role": "assistant", "label": "refined_solution", "content": refined_solution}]
                
                sample = {
                    "timeout": 30,
                    "case_count": 32,
                    "test_case_generator": test_case_generator.strip(),
                    "canonical_solution": refined_solution.strip(),
                    "entry_point": entry_point,
                }
                
                result = self.sb.run_sample(sample)
                status = result["status"]
                traceback = result["traceback"]
                
                if result["status"] == "success": break
                
                refine_loop_count -= 1
            data += [{"messages": messages, "status": status}]
            
        ds = Dataset.from_pandas(pd.DataFrame(data=data))
        ds.push_to_hub("Elfsong/Afterburner_code_correction", self.ds_name)
                
if __name__ == "__main__":
    code_corrector = Code_Corrector()
    code_corrector.pipeline()
                
                
                
                
            
            
            

                