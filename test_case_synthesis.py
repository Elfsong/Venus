# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 21

import os
from datasets import load_dataset
import src.prompts as prompts
from src.utils import OpenAIClient


class TestCasesSynthesizer:
    def __init__(self, lang):
        self.lang = lang
        self.ds = load_dataset("Elfsong/venus", self.lang)
        self.client = OpenAIClient(model_name="gpt-4o", api_key=os.getenv("CLIENT_API_KEY"))
        

    def generate_test_cases(self, instance):
        problem_description = instance['content']
        code_prompt = ""
        
        if instance['codeSnippets']:
            for prompt_obj in instance['codeSnippets']:
                if prompt_obj['langSlug'] == self.lang:
                    code_prompt = prompt_obj['code']
                
        messages = [
            {"role": "system", "content": "You are a code expert."},
            {"role": "user", "content": prompts.solution_generation.format(problem_description=problem_description, lang=self.lang, code_prompt=code_prompt)}
        ]
        response = self.client.inference(messages)
        return response['solution']
        pass


if __name__ == "__main__":
    test_cases_synthesizer = TestCasesSynthesizer("python")
    