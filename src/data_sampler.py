# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/08/31

import os
import random
import pandas as pd
from tqdm import tqdm
from utils import OpenAIClient
from datasets import load_dataset


class Data_Synthesizer():
    def __init__(self, generation_count) -> None:
        self.seed = 42
        self.streaming = False
        self.model_name = "gpt-4o"
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
            Drawing inspiration from the following code snippets, generate a challenging Python coding question. Define the input format and expected output format clearly. Return in this JSON format:
            Response: {"problem_description": "<problem_description>", "canonical_solution": "<canonical_solution>", "test_case_generator": "<test_case_generator>"}
        """
        messages = [
            {"role": "system", "content": "You are a code expert. "},
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
    
    def code_sample(self, code):
        code = code["content"].split("\n")
        start, end = 0, len(code)
        if len(code) > 5:
            start = random.randrange(len(code)-5)
            end = start + random.randrange(5, 10)
        return "\n".join(code[start:end])
    
    def pipeline(self):
        for _ in tqdm(range(self.generation_count)):
            seeds = ""
            for seed in self.seed_mix():
                code = self.code_sample(seed)
                seeds += code
            response = self.synthesis(seeds=seeds)
            print(response)
    
    
if __name__ == "__main__":
    data_synthesizer = Data_Synthesizer(generation_count=10)
    data_synthesizer.pipeline()
    