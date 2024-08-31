# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/08/31

import random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


class Data_Synthesizer():
    def __init__(self, generation_count) -> None:
        self.seed = 42
        self.streaming = False
        self.generation_count = generation_count
        self.languages = ["python", "c", "cpp", "html"]
        
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
        pass
    
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
            seeds = self.seed_mix()
            for seed in seeds:
                code = self.code_sample(seed)
                print(code)
                print("========================")
    
    
if __name__ == "__main__":
    data_synthesizer = Data_Synthesizer(generation_count=10)
    data_synthesizer.pipeline()
    