# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/08/31

import os
import re
import uuid
import pickle
import random
import sandbox
import platform
import traceback
import faulthandler
import pandas as pd
from tqdm import tqdm
from utils import OpenAIClient
from datasets import Dataset, load_dataset


class Data_Synthesizer():
    def __init__(self, generation_count) -> None:
        self.seed = 42
        self.streaming = False
        self.model_name = "gpt-4o"
        self.ds_name = str(uuid.uuid1())
        self.model_token = os.getenv("OPENAI_TOKEN")
        self.generation_count = generation_count
        self.languages = ["python", "c", "cpp", "html"]
        self.openai_client = OpenAIClient(model_name=self.model_name, model_token=self.model_token)
        
        print("Loading datasets...")
        self.data_sources = dict()
        for language in self.languages:
            self.data_sources[language] = self.get_dataset(language=language)
        
    def get_dataset(self, language):
        ds = load_dataset(
            "bigcode/starcoderdata", 
            data_dir=language, 
            split="train", 
            streaming=self.streaming, 
            cache_dir="/mnt/disks/venus_data"
        ).shuffle(seed=self.seed)
        return iter(ds)
    
    def synthesis(self, seeds):
        prompt = """
Using the provided code snippets as a reference, create a challenging and realistic coding challenge that is not specific to any programming language.

For each 'input = generate_test_case()', the corresponding output must be 'output = <entry_point>(*input)'. If the entry point function takes a single argument, wrap the input in a tuple to maintain consistency. For functions with multiple arguments, ensure the inputs are encapsulated within a tuple.

Return the result in the following JSON format:
                        
Return in this JSON format:
{
    "problem_description": "a markdown problem description with input/output examples and constraints",
    "canonical_solution": "a Python function that accepts test case input and returns the expected test case output",
    "simple_test_case_generator": "a Python function 'generate_test_case()' to randomly return a test case input. The random range can be limited to a reasonable test range.",
    "full_test_case_generator": "a Python function 'generate_test_case()' to randomly return a test case input. The random range should cover the full range.",
    "entry_point": "the entry point of the canonical_solution"
}
        """
        messages = [
            {"role": "system", "content": "You are a code expert. Generate pure code for code fields."},
            {"role": "user", "content": prompt + f"code snippets: {seeds}"}
        ]
        response = self.openai_client.inference(messages)
        return response
    
    def seed_mix(self):
        seeds = list()
        while not seeds:
            for lang in self.data_sources:
                instance = next(self.data_sources[lang])
                if bool(random.getrandbits(1)):
                    seeds += [instance]
        return seeds
    
    @staticmethod
    def code_sample(code):
        code = code["content"].split("\n")
        start, end = 0, len(code)
        if len(code) > 5:
            start = random.randrange(len(code)-5)
            end = start + random.randrange(5, 10)
        return "\n".join(code[start:end])
    
    
    def pipeline(self):
        data = list()
        sb = sandbox.Sandbox()
        
        for _ in tqdm(range(self.generation_count)):
            try:
                seeds = ""
                for seed in self.seed_mix():
                    code = Data_Synthesizer.code_sample(seed)
                    seeds += code
                response = self.synthesis(seeds=seeds)
                
                sample = {
                    "timeout": 30,
                    "case_count": 32,
                    "test_case_generator": response["simple_test_case_generator"].strip(),
                    "canonical_solution": response['canonical_solution'].strip(),
                    "entry_point": response['entry_point'].strip(),
                }
                
                result = sb.run_sample(sample)
                
                response["problem_description"] = str(response["problem_description"])
                response["canonical_solution"] = str(response["canonical_solution"])
                response["simple_test_case_generator"] = str(response["simple_test_case_generator"])
                response["full_test_case_generator"] = str(response["full_test_case_generator"])
                response["cases"] = str(result["cases"])
                response["traceback"] = str(result["traceback"])
                response["status"] = str(result["status"])
                
                data += [response]
            except Exception as e:
                print(f"Error and skipped: {e}")
        
        ds = Dataset.from_pandas(pd.DataFrame(data=data))
        ds.push_to_hub("Elfsong/Afterburner", self.ds_name)

if __name__ == "__main__":
    data_synthesizer = Data_Synthesizer(generation_count=1000)
    data_synthesizer.pipeline()
    