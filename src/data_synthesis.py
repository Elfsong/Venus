# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/04

import os
import uuid
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
        self.ds_name = str(uuid.uuid1())
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
            test_case_generator = response["test_case_generator"].strip()
            
            sample = {
                "timeout": 64,
                "case_count": 64,
                "test_case_generator": test_case_generator,
                "solution": solution,
                "entry_point": "solution",
            }
            result = self.sandbox.run_generation(sample)
            
            instance["meta_info"]["problem_description"] = problem_description
            instance["meta_info"]["test_case_generator"] = test_case_generator
            instance["meta_info"]["test_cases"] = result["cases"]
            
            instance["solutions"] += [{
                "code": solution, "status": result["status"], "traceback": result["traceback"],
                "code_time": result["code_time"], "code_mem": result["code_mem"]
            }]
            
            problem_description = instance["meta_info"]["problem_description"]
            test_case_generator = instance["meta_info"]["test_case_generator"]
            last_solution = instance["solutions"][-1]
            instance['messages'] += [{"role": "system", "content": prompts.solution_generation_system_prompt}]
            instance['messages'] += [{"role": "user", "content": prompts.solution_generation_user_prompt.format(problem_description=problem_description, test_case_generator=test_case_generator)}]
            instance['messages'] += [{"role": "assistant", "content": last_solution['code']}]
            
            status = True
        except Exception as e:
            print(f"Error@problem_generation: {e}")
        return status
    
    def code_correction(self, instance):
        status = False
        try:
            traceback = instance["solutions"][-1]['traceback']
            instance['messages'] += [{"role": "user", "content": prompts.solution_correction_user_prompt.format(traceback=traceback)}]
            response = self.openai_client.inference(instance['messages'])
            
            sample = {
                "timeout": 30,
                "case_count": 128,
                "test_case_generator": instance["meta_info"]["test_case_generator"],
                "solution": response["corrected_solution"].strip(),
            }
            result = self.sandbox.run_generation(sample)
            
            corrected_solution = response["corrected_solution"].strip()
            instance['messages'] += [{"role": "assistant", "content": corrected_solution}]
            instance["solutions"] += [{
                "code": corrected_solution, "status": result["status"], 
                "traceback": result["traceback"], 
                "code_time": result["code_time"], "code_mem": result["code_mem"]
            }]

            status = True
        except Exception as e:
            print(f"Error@solution_correction: {e}")
        return status
    
    def instruction_generation(self, instance, instruction):
        status = False
        try:
            instance['messages'] += [{"role": "user", "content": prompts.instruction_generation_user_prompt.format(instruction=instruction)}]
            response = self.openai_client.inference(instance['messages'])
            
            optimized_solution = response["optimized_solution"].strip()
            
            sample = {
                "timeout": 128,
                "case_count": 32,
                "test_case_generator": instance["meta_info"]["test_case_generator"],
                "original_solution": instance["solutions"][-1]['code'],
                "optimized_solution": optimized_solution,
            }
            
            result = self.sandbox.run_evaluation(sample)
            
            instance['messages'] += [{"role": "assistant", "content": optimized_solution}]
            instance["solutions"] += [{
                "code": optimized_solution, "status": result["status"], 
                "traceback": result["traceback"], 
                "code_time": result["code_time"], "code_mem": result["code_mem"]
            }]
            
            status = True
        except Exception as e:
            print(f"Error@instruction_generation: {e}")
        return status
    
    
    def pipeline(self, loop=3):
        instance = dict(messages=list(), meta_info=dict(), solutions=list())

        # Code Mixture
        if not self.seed_mixture(instance): return None
        
        # Problem Synthesis
        if not self.problem_synthesis(instance, "Python"): return None
        
        # Optimization 
        for _ in tqdm(range(loop), desc="Optimizing..."):
            last_solution = instance["solutions"][-1]
            if last_solution['status'] != "success":
                self.code_correction(instance)
            else:
                instruction = random.choice(["Generate a solution that optimizes execution speed", "Generate a solution that optimizes memory usage"])
                self.instruction_generation(instance, instruction)
        
        # Statistic
        success_count = 0
        status_list = list()
        for solution in instance["solutions"]:
            if solution['status'] == "success":
                success_count += 1
            status_list += [solution['status']]
        instance['success_count'] = success_count
        instance['status_list'] = status_list
        instance['meta_info']['success_count'] = success_count
        
        return instance
        
    def run(self):
        instances = list()
        for _ in tqdm(range(self.generation_count)):
            instance = self.pipeline(loop=5)
            if instance:
                instances.append(instance)
            
        ds = Dataset.from_pandas(pd.DataFrame(data=instances))
        ds.push_to_hub("Elfsong/Afterburner_RAW", self.ds_name)
    
if __name__ == "__main__":
    for index in tqdm(range(20)):
        try:
            data_synthesis = Data_Synthesis(model_name="gpt-4o", language_source=["python", "cpp", "c"], generation_count=20)
            # data_synthesis = Data_Synthesis(model_name="gpt-4o", language_source=["python"], generation_count=20)
            print("Current UUID: ", data_synthesis.ds_name)
            data_synthesis.run()
        except Exception as e:
            print(e)
            
        