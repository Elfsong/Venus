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

class Efficient_Optimizer():
    def __init__(self) -> None:
        self.datasets = load_dataset("Elfsong/Afterburner_code_correction", "469c00b0-6c2d-11ef-a1aa-42010a94000c")["train"]
        self.ds_name = str(uuid.uuid1())
        self.optimization_loop_count = 2
        self.model_name = "gpt-4o"
        self.model_token = os.getenv("OPENAI_TOKEN")
        self.sandbox = sandbox.Sandbox()
        self.openai_client = OpenAIClient(model_name=self.model_name, model_token=self.model_token)
    
    def solution_optimize(self, messages, objective):
        prompt = f"""
            Generate a {objective} solution and return the optimized solution in the specified JSON format.

            Expected JSON Format:
            {{
                "optimized_solution": "The optimized solution that accepts the test case input and returns the expected test case output"
            }}
        """
        
        messages += [
            {"role": "user", "label": "optimization", "content": prompt}
        ]
        response = self.openai_client.inference(messages)
        return response
    
    def pipeline(self) -> None:
        data = list()
        
        for instance in tqdm(self.datasets):            
            messages = instance["messages"]
            test_case_generator = instance["test_case_generator"]
            entry_point = instance["entry_point"]

            try:               
                result = self.solution_optimize(messages, "faster")
                optimized_solution = result["optimized_solution"].strip()
                
                sample = {
                    "timeout": 30,
                    "case_count": 32,
                    "test_case_generator": test_case_generator,
                    "canonical_solution": optimized_solution,
                    "entry_point": entry_point,
                }
                
                result = self.sandbox.run_sample(sample)
                status = result["status"]
                traceback = result["traceback"]
                code_time = result["code_time"]
                code_mem = result["code_mem"]
                
                messages += [{"role": "assistant", "label": "solution", "content": optimized_solution, "status": status, "traceback": traceback, "code_time": code_time, "code_mem": code_mem,}]
            except Exception as e:
                print(f"Error: {e}")
            
            data += [{
                "messages": messages, 
                "entry_point": entry_point,
                "test_case_generator": test_case_generator, 
            }]
            
        ds = Dataset.from_pandas(pd.DataFrame(data=data))
        ds.push_to_hub("Elfsong/Afterburner_code_optimization", self.ds_name)

            
if __name__ == "__main__":
    efficient_optimizer = Efficient_Optimizer()
    efficient_optimizer.pipeline()