# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 21

import os
import argparse
from tqdm import tqdm
import src.prompts as prompts
from datasets import load_dataset, Dataset
from src.utils import OpenAIClient


class TestCasesSynthesizer:
    def __init__(self, lang):
        self.lang = lang
        self.ds = load_dataset("Elfsong/venus", self.lang, download_mode="force_redownload")
        self.client = OpenAIClient(model_name="gpt-4o", model_token=os.getenv("CLIENT_API_KEY"))

    def generate_test_case(self, instance, canonical_solution):
        problem_description = instance['content']
                        
        messages = [
            {"role": "system", "content": "You are a code expert."},
            {"role": "user", "content": prompts.case_generation.format(problem_description=problem_description, canonical_solution=canonical_solution, lang=self.lang)}
        ]
        response = self.client.inference(messages)
        return response
    
    def function_validation(self, canonical_solution, test_case_functions):
        namespace = dict()
        
        exec("import json", namespace)
        exec("import random", namespace)
        exec("import collections", namespace)
        exec("from json import loads", namespace)
        exec("from typing import Tuple, List", namespace)
        exec("from collections import defaultdict", namespace)
        
        exec(canonical_solution, namespace)
        exec(test_case_functions['serialize_input'], namespace)
        exec(test_case_functions['deserialize_input'], namespace)
        exec(test_case_functions['serialize_output'], namespace)
        exec(test_case_functions['deserialize_output'], namespace)
        exec(test_case_functions['generate_test_case_input'], namespace)
        
        exec("test_case_input = generate_test_case_input()", namespace)
        exec("test_case_input_serialized = serialize_input(test_case_input)", namespace)
        exec("test_case_input = deserialize_input(test_case_input_serialized)", namespace)
        exec("solution = Solution()", namespace)
        exec("test_case_output = solution.{}(*test_case_input)".format(test_case_functions['entry_point']), namespace)
        exec("test_case_output_serialized = serialize_output(test_case_output)", namespace)
        exec("test_case_output = deserialize_output(test_case_output_serialized)", namespace)
        print("Bingo 😀")
    
    def pipeline(self):
        new_dl = list()
        for instance in self.ds['train'].to_list()[:10]:
            print("[+] Generating test cases for {id}.{name}".format(id=instance['question_id'], name=instance['name']))
            
            try:
                for s in instance['rt_list']:
                    canonical_solution = s['code']
                    if 'stdin' not in canonical_solution:
                        break
                
                test_case_functions = self.generate_test_case(instance, canonical_solution)
                
                self.function_validation(canonical_solution, test_case_functions)
                instance['test_case_functions'] = test_case_functions
                new_dl.append(instance)
            except Exception as e:
                print(f"[-] Failed to parse test cases for {instance['name']}. Error: {e}")
                continue
        
        new_ds = Dataset.from_list(new_dl)
        new_ds.push_to_hub(f"Elfsong/Venus_Case", self.lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default="python3") 
    args = parser.parse_args()

    test_cases_synthesizer = TestCasesSynthesizer(args.language)
    test_cases_synthesizer.pipeline()
    