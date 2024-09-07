# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/04

import os
import random
import prompts
import sandbox
import pandas as pd
from tqdm import tqdm
from utils import OpenAIClient
from datasets import Dataset, load_dataset

class Data_Synthesis:
    def __init__(self, model_name, language_source, generation_count) -> None:
        # Pipeline Config
        self.generation_count = generation_count
        self.sandbox = sandbox.Sandbox()
        
        # OpenAI Client
        self.model_name = model_name
        self.model_token = os.getenv("OPENAI_TOKEN")
        self.openai_client = OpenAIClient(model_name=self.model_name, model_token=self.model_token)
        
        # Source Datasets
        self.data_sources = dict()
        self.languages = language_source
        for language in self.languages:
            self.data_sources[language] = Data_Synthesis.get_dataset(language=language)
            
    @staticmethod
    def get_dataset(language, streaming=False, seed=42):
        ds = load_dataset(
            "bigcode/starcoderdata", 
            data_dir=language, split="train", 
            cache_dir="/mnt/disks/venus_data",
            streaming=streaming, 
        ).shuffle(seed=seed)
        return iter(ds)
    
    @staticmethod
    def code_sample(code, l=5, r=15):
        code = code["content"].split("\n")
        start, end = 0, len(code)
        if len(code) > l:
            start = random.randrange(len(code)-l)
            end = start + random.randrange(l, r)
        return "\n".join(code[start:end])
    
    def seed_mixture(self, instance):
        status = False
        code_reference = str()
        try:
            while not code_reference:
                for lang in self.data_sources:
                    code = next(self.data_sources[lang])
                    if bool(random.getrandbits(1)): 
                        code = Data_Synthesis.code_sample(code)
                        code_reference += f'Code Snippet:\n{code}\n\n'
            instance["code_reference"] = code_reference
            status = True
        except Exception as e:
            print(f"Error@seed_mixture: {e}")
        return status
    
    def problem_synthesis(self, instance, language):
        status = False
        try:
            messages = [
                {"role": "system", "content": prompts.problem_synthesis_system_prompt},
                {"role": "user", "content": prompts.problem_synthesis_user_prompt.format(code_reference=instance['code_reference'], language=language)}
            ]
            response = self.openai_client.inference(messages)
            
            solution = response["canonical_solution"].strip()
            problem_description = response["problem_description"].strip()
            simple_test_case_generator = response["simple_test_case_generator"].strip()
            full_test_case_generator = response["full_test_case_generator"].strip()
            
            sample = {
                "timeout": 30,
                "case_count": 128,
                "test_case_generator": simple_test_case_generator,
                "solution": solution,
                "entry_point": "solution",
            }
            result = self.sandbox.run_generation(sample)
            
            instance["meta_info"]["problem_description"] = problem_description
            instance["meta_info"]["simple_test_case_generator"] = simple_test_case_generator
            instance["meta_info"]["full_test_case_generator"] = full_test_case_generator
            instance["meta_info"]["test_cases"] = result["cases"]
            
            instance["solutions"] += [{
                "code": solution, "status": result["status"], "traceback": result["traceback"],
                "code_time": result["code_time"], "code_mem": result["code_mem"]
            }]
            
            status = True
        except Exception as e:
            print(f"Error@problem_generation: {e}")
        return status
    
    def code_correction(self, instance):
        pass
    
    def solution_generation(self, instance, language, instruction):
        status = False
        try:
            messages = [
                {"role": "system", "content": prompts.solution_generation_system_prompt},
                {"role": "user", "content": prompts.solution_generation_user_prompt.format(
                    problem_description=instance["meta_info"]["problem_description"], 
                    test_case_generator=instance["meta_info"]["simple_test_case_generator"], 
                    language=language, instruction=instruction)
                }
            ]
            response = self.openai_client.inference(messages)
            
            instance["meta_info"]["solution"] = response["canonical_solution"].strip()
            instance["meta_info"]["entry_point"] = response["entry_point"].strip()
            instance["messages"] += messages
            
            sample = {
                "timeout": 30,
                "case_count": 128,
                "test_case_generator": instance["meta_info"]["simple_test_case_generator"],
                "solution": instance["meta_info"]["solution"],
                "entry_point": instance["meta_info"]["entry_point"],
            }
            
            result = self.sandbox.run_generation(sample)
            print(str(result["status"]))

            status = True
        except Exception as e:
            print(f"Error@solution_generation: {e}")
        return status
    
    
    def pipeline(self):
        instance = dict(messages=list(), meta_info=dict(), solutions=list())

        # Code Mixture
        if not self.seed_mixture(instance): return None
        
        # Problem Synthesis
        if not self.problem_synthesis(instance, "Python"): return None
        
        # Solution Generation
        # if not self.solution_generation(instance, "Python", prompts.normal_instruction): return None
        
        
        while True:
            if instance["solutions"][-1]['status'] != "success":
                self.code_correction(instance)
            else:
                self.solution_generation(instance)
        
        return instance
        
    def run(self):
        instances = list()
        for _ in tqdm(range(self.generation_count)):
            instance = self.pipeline()
            if instance:
                instances.append(instance)
    
if __name__ == "__main__":
    data_synthesis = Data_Synthesis(model_name="gpt-4o", language_source=["python"], generation_count=10)
    data_synthesis.run()
        