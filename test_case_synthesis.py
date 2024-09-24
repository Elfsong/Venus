# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 21

import os
import uuid
import random
import argparse
from tqdm import tqdm
import src.prompts as prompts
import src.sandbox as sandbox
from src.utils import OpenAIClient
from datasets import load_dataset, Dataset


class TestCasesSynthesizer:
    def __init__(self, lang):
        self.lang = lang
        self.sandbox = sandbox.Sandbox()
        self.from_ds = load_dataset("Elfsong/venus", self.lang, download_mode="force_redownload")
        self.to_ds = load_dataset("Elfsong/venus_case", self.lang, download_mode="force_redownload")
        self.to_ds_id = set([i['question_id'] for i in self.to_ds['train']])
        self.dl = self.from_ds['train'].to_list()
        self.client = OpenAIClient(model_name="gpt-4o", model_token=os.getenv("CLIENT_API_KEY"))

    def generate_test_case(self, instance, canonical_solution):
        problem_description = instance['content']
                        
        messages = [
            {"role": "system", "content": "You are a code expert."},
            {"role": "user", "content": prompts.case_generation.format(problem_description=problem_description, canonical_solution=canonical_solution, lang=self.lang)}
        ]
        response = self.client.inference(messages)
        return response
    
    def function_validation(self, solutions, test_case_functions):
        namespace = dict()
        
        exec("import re", namespace)
        exec("import sys", namespace)
        exec("import json", namespace)
        exec("import math", namespace)
        exec("import copy", namespace)
        exec("import heapq", namespace)
        exec("import heapq", namespace)
        exec("import bisect", namespace)
        exec("import string", namespace)
        exec("import string", namespace)
        exec("import random", namespace)
        exec("import itertools", namespace)
        exec("import functools", namespace)
        exec("import collections", namespace)
        exec("from json import loads", namespace)
        exec("from sys import maxsize, stdin", namespace)
        exec("from functools import lru_cache, cache", namespace)
        exec("from typing import List, Optional, Tuple", namespace)
        exec("from typing import Tuple, List, Optional", namespace)
        exec("from heapq import heappush, heappop, heapify", namespace)
        exec("from bisect import bisect_left, bisect_right", namespace)
        exec("from itertools import permutations, zip_longest", namespace)
        exec("from math import floor, ceil, factorial, sqrt, inf", namespace)
        exec("from collections import defaultdict, Counter, deque", namespace)
        exec("from collections import deque, defaultdict, OrderedDict", namespace)
        
        exec("class ListNode:\n\tdef __init__(self, val=0, next=None):\n\t\tself.val=val\n\t\tself.next=next", namespace)
        exec("class TreeNode:\n\tdef __init__(self, val=0, left=None, right=None):\n\t\tself.val=val\n\t\tself.left=left\n\t\tself.right=right", namespace)
        
        test_cases = list()
        canonical_solution = random.choice(solutions)
        try:
            exec(canonical_solution, namespace)
            exec(test_case_functions['serialize_input'], namespace)
            exec(test_case_functions['deserialize_input'], namespace)
            exec(test_case_functions['serialize_output'], namespace)
            exec(test_case_functions['deserialize_output'], namespace)
            exec(test_case_functions['generate_test_case_input'], namespace)

            for _ in range(64):
                exec("test_case_input = generate_test_case_input()", namespace)
                exec("test_case_input_serialized = serialize_input(test_case_input)", namespace)
                exec("test_case_input = deserialize_input(test_case_input_serialized)", namespace)
                exec("solution = Solution()", namespace)
                exec("test_case_output = solution.{}(*test_case_input)".format(test_case_functions['entry_point']), namespace)
                exec("test_case_output_serialized = serialize_output(test_case_output)", namespace)
                exec("test_case_output = deserialize_output(test_case_output_serialized)", namespace)
                
                for index in range(min(32, len(solutions))):
                    exec("solution = Solution()", namespace)
                    exec("test_case_output_ = solution.{}(*test_case_input)".format(test_case_functions['entry_point']), namespace)
                    exec("test_case_output_serialized_ = serialize_output(test_case_output)", namespace)
                    exec("test_case_output_ = deserialize_output(test_case_output_serialized)", namespace)
                    if namespace['test_case_output'] != namespace['test_case_output_']:
                        raise Exception(f"Test case output mismatch")
                
                print("✅", end=" ", flush=True)
                test_cases.append({"input": namespace['test_case_input_serialized'], "output": namespace['test_case_output_serialized']})
        except Exception as e:
            print(f"\n🔴 Validation Failed: {e}")
            return False
            
        print("\n🟢 Validation Passed")
        return True
    
    def pipeline(self, sub_dl):
        new_dl = list()
       
        for index, instance in enumerate(sub_dl):
            print("========== Generating test cases for {id}.{name} [{index}/{total}]".format(id=instance['question_id'], name=instance['name'], index=index+1, total=10))
            
            if instance['question_id'] in self.to_ds_id:
                print(f"Found the case in the dataset. Skipped ✅")
                continue
            
            try:
                # 1. Select the canonical solution
                solutions = instance['rt_list'] + instance['mm_list']
                solution_candidates = list()
                for solution in solutions:
                    code = solution['code']
                    if 'stdin' not in code:
                        solution_candidates += [code]
                
                if len(solution_candidates) == 0:
                    print(f"[-] No canonical solution found")
                    print(f"🔴 Failed")
                    continue
                print(f"[+] Found [{len(solution_candidates)}] solutions")
                
                # 2. Select a canonical solution
                canonical_solution = random.choice(solution_candidates)
                
                # 3. Generate test cases
                print(f"[+] Generating test cases...")
                test_case_functions = self.generate_test_case(instance, canonical_solution)
                
                # 4. Validate the test cases
                print(f"[+] Validating test cases...")
                sample = {
                    "timeout": 120, 
                    "test_case_functions": test_case_functions,
                    "solutions": solution_candidates,
                }
                result = self.sandbox.run_test_case_validation(sample)
                
                # 5. Save the test cases
                if result['status'] == "success":
                    instance['test_case_functions'] = test_case_functions
                    instance['test_cases'] = result['test_cases']
                    new_dl.append(instance)
                    print(f"🟢 Success")
                else:
                    print(f"🔴 Failed: ", result['status'])
                
            except Exception as e:
                print(f"🔴 Failed: ", e)
                continue
        
        new_ds = Dataset.from_list(new_dl)
        ds_name = str(uuid.uuid1())
        new_ds.push_to_hub(f"Elfsong/Venus_Case_Temp", ds_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default="python3") 
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    args = parser.parse_args()

    test_cases_synthesizer = TestCasesSynthesizer(args.language)
    
    for i in tqdm(range(args.start, args.end)):
        test_cases_synthesizer.pipeline(test_cases_synthesizer.dl[i*10:(i+1)*10])
    