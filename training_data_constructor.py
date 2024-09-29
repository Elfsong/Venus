# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-09-24

import json
import random
from src import prompts
from tqdm import tqdm
from datasets import load_dataset, Dataset

class DataConstructor:
    def __init__(self, lang, metric):
        self.lang = lang
        self.metric = metric
        assert self.lang in ['python3', 'cpp']
        assert self.metric in ['runtime', 'memory']
        self.ds = load_dataset("Elfsong/Venus", self.lang, download_mode="force_redownload")
        
    def distribution_constructor(self, distribution):
        quartile = {"q1": [], "q2": [], "q3": [], "q4": []}
        cumulative_percentile = 0.0

        for metric, percentile in distribution:
            if cumulative_percentile < 25:
                quartile['q1'].append(metric)
            elif cumulative_percentile < 50:
                quartile['q2'].append(metric)
            elif cumulative_percentile < 75:
                quartile['q3'].append(metric)
            else:
                quartile['q4'].append(metric)
            cumulative_percentile += percentile

        return quartile
    
    def pcd_instance_constructor(self, instance, runtime_gap, memory_gap):
        solutions = instance['rt_list'] if self.metric == 'runtime' else instance['mm_list']
        solutions = [solution for solution in solutions if 'stdin' not in solution['code'] and 'Solution' in solution['code']]
            
        if len(solutions) < 2: return None
    
        solution_c = random.choice(solutions)
        solution_r = random.choice(solutions)
        
        # Select solution_choose and solution_reject      
        if int(solution_c[self.metric]) <= int(solution_r[self.metric]) - (runtime_gap if self.metric == 'runtime' else memory_gap):
            answers = random.choice(["A", "B", "C"])
            
            # pick an irrelevant solution
            r = random.randint(0, len(self.ds['train']) - 1)
            random_instance = self.ds['train'][r]
            random_solution = random.choice(random_instance['rt_list'] + random_instance['mm_list'])
            
            options = [solution_r, random_solution]
            random.shuffle(options)
            if answers == "A":
                options.insert(0, solution_c)
            elif answers == "B":
                options.insert(1, solution_c)
            else:
                options.insert(2, solution_c)
            
            pcd_instance = {
                'id': instance['question_id'],
                'conversations': [
                    {
                        'from': 'human',
                        'value': prompts.PCD_prompt.format(problem_description=instance['content'], metric=self.metric, solution_a=options[0]['code'], solution_b=options[1]['code'], solution_c=options[2]['code'])
                    },
                    {
                        'from': 'gpt',
                        'value': f"Solution {answers}"
                    }
                ],
                "correct_answer": answers,
                'system': f"You are a code expert that identifies the most {self.metric} efficient solution from a given problem description and corresponding solutions.",
            }
            return pcd_instance
        else:
            return None
    
    def profile_comparison_data_constructor(self, pair_num=200, runtime_gap=10, memory_gap=100):
        # [problem_description, metric, solution_c, solution_r] → [solution_c]       
        data_collection = list()
        
        for instance in tqdm(self.ds['train']):
            for _ in range(pair_num):
                pcd_instance = self.pcd_instance_constructor(instance, runtime_gap, memory_gap)
                if pcd_instance is not None:
                    data_collection.append(pcd_instance)
                    
        print(f"PCD [{self.lang}-{self.metric}] Size: {len(data_collection)}")
        pcd = Dataset.from_list(data_collection)
        pcd.push_to_hub(f"Elfsong/Venus_PCD", f'{self.lang}-{self.metric}')
    
    def profile_optimization_data_constructor(self, pair_num=200, runtime_gap=10, memory_gap=100):
        # [problem_description, instruction, solution_r] → [solution_c]
        
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
                            'instruction': self.metric,
                            'solution_c': solution_c,
                            'solution_r': solution_r
                        }
                    ]
        print(f"POD [{self.lang}-{self.metric}] Size: {len(data_collection)}")
        pcd = Dataset.from_list(data_collection)
        pcd.push_to_hub(f"Elfsong/Venus_POD", f'{self.lang}-{self.metric}')
    
    def controllable_code_generation_data_constructor(self):
        # [problem_description, instruction] → [solution_c]
        data_collection = list()
        
        for instance in tqdm(self.ds['train']):
            distribution = json.loads(instance['runtimeDistribution'])['distribution'] if self.metric == 'runtime' else json.loads(instance['memoryDistribution'])['distribution']
            quartile = self.distribution_constructor(distribution)
            solutions = instance['rt_list'] if self.metric == 'runtime' else instance['mm_list']
            solutions = [solution for solution in solutions if 'stdin' not in solution['code'] and 'Solution' in solution['code']]
            
            for solution in solutions:
                label = 0
                if len(quartile['q1']) > 0 and int(solution[self.metric]) <= int(quartile['q1'][-1]):
                    label = 0
                elif len(quartile['q2']) > 0 and int(solution[self.metric]) <= int(quartile['q2'][-1]):
                    label = 1
                elif len(quartile['q3']) > 0 and int(solution[self.metric]) <= int(quartile['q3'][-1]):
                    label = 2
                else:
                    label = 3
                    
                data_collection += [{
                    'id': instance['question_id'],
                    'problem_description': instance['content'],
                    'instruction': label,
                    'solution_c': solution,
                }]
        
        print(f"CCG [{self.lang}-{self.metric}] Size: {len(data_collection)}")
        pcd = Dataset.from_list(data_collection)
        pcd.push_to_hub(f"Elfsong/Venus_CCG", f'{self.lang}-{self.metric}')
    
    
if __name__ == "__main__":
    data_constructor = DataConstructor("cpp", 'runtime')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=10, memory_gap=100)
    # data_constructor.profile_optimization_data_constructor(pair_num=200, runtime_gap=10, memory_gap=100)
    # data_constructor.controllable_code_generation_data_constructor()
    
    data_constructor = DataConstructor("cpp", 'memory')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=10, memory_gap=100)
    # data_constructor.profile_optimization_data_constructor(pair_num=200, runtime_gap=10, memory_gap=100)
    # data_constructor.controllable_code_generation_data_constructor()
    
    data_constructor = DataConstructor("python3", 'runtime')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=20, memory_gap=200)
    # data_constructor.profile_optimization_data_constructor(pair_num=200, runtime_gap=20, memory_gap=200)
    # data_constructor.controllable_code_generation_data_constructor()
    
    data_constructor = DataConstructor("python3", 'memory')
    data_constructor.profile_comparison_data_constructor(pair_num=200, runtime_gap=20, memory_gap=200)
    # data_constructor.profile_optimization_data_constructor(pair_num=200, runtime_gap=20, memory_gap=200)
    # data_constructor.controllable_code_generation_data_constructor()
        