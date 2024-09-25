# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-09-24

import random
from tqdm import tqdm
from datasets import load_dataset, Dataset

class DataConstructor:
    def __init__(self, lang, metric):
        self.lang = lang
        self.metric = metric
        assert self.lang in ['python3', 'cpp']
        assert self.metric in ['runtime', 'memory']
        self.ds = load_dataset("Elfsong/Venus", self.lang, download_mode="force_redownload")
        
    def profile_comparison_data_constructor(self, pair_num=200, runtime_gap=10, memory_gap=100):
        # [problem_description, metric, solution_c, solution_r] → [solution_c]       
        data_collection = list()
        
        for instance in tqdm(self.ds['train']):
            solutions = instance['rt_list'] if self.metric == 'runtime' else instance['mm_list']
            solutions = [solution for solution in solutions if 'stdin' not in solution['code'] and 'Solution' in solution['code']]
            
            if len(solutions) < 2: continue
                
            for _ in range(pair_num):
                solution_c = random.choice(solutions)
                solution_r = random.choice(solutions)
                if int(solution_c[self.metric]) <= int(solution_r[self.metric]) + (runtime_gap if self.metric == 'runtime' else memory_gap):
                    data_collection += [
                        {
                            'id': instance['question_id'],
                            'problem_description': instance['content'],
                            'metric': self.metric,
                            'solution_c': solution_c,
                            'solution_r': solution_r
                        }
                    ]
        print("PCD Size:", len(data_collection))
        pcd = Dataset.from_list(data_collection)
        pcd.push_to_hub(f"Elfsong/Venus_PCD", f'{self.lang}-{self.metric}')
    
    def profile_optimization_data_constructor(self):
        # [problem_description, instruction, solution_r] → [solution_c]
        pass
    
    def controllable_code_generation_data_constructor(self):
        # [problem_description, instruction] → [solution_c]
        pass
    
    
if __name__ == "__main__":
    data_constructor = DataConstructor("cpp", 'runtime')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=10, memory_gap=100)
    
    data_constructor = DataConstructor("cpp", 'memory')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=10, memory_gap=100)
    
    data_constructor = DataConstructor("python3", 'runtime')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=20, memory_gap=200)
    
    data_constructor = DataConstructor("python3", 'memory')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=20, memory_gap=200)
        
        