# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 14

import os
import time
import json
import uuid
import requests
import argparse
import datasets
import traceback
import pandas as pd
from tqdm import tqdm
import src.prompts as prompts
from datasets import Dataset, load_dataset
from datasets import get_dataset_config_names
from src.utils import OpenAIClient, retry, vital_retry


class LeetCodeOperation:
    def __init__(self, lang, mode) -> None:
        self.lang = lang
        self.mode = mode
        self.instances = list()
        self.existing_question_ids = set()
        self.url = "https://leetcode.com/graphql/"
        self.model_token = os.getenv("CLIENT_API_KEY")
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE")
        self.leetcode_crsf_token = os.getenv("LEETCODE_CRSF_TOKEN")
        self.lang_code_mapping = {
            "cpp": 0, "java": 1, "python": 2, "python3": 11, "mysql": 3, "mssql": 14, "oraclesql": 15, "c": 4,
            "csharp": 5, "javascript": 6, "typescript": 20, "bash": 8, "php": 19, "swift": 9, "kotlin": 13, "dart": 24,
            "golang": 10, "ruby": 7, "scala": 12, "html": 16, "pythonml": 17, "rust": 18, "racket": 21, "erlang": 22,
            "elixir": 23, "pythondata": 25, "react": 26, "vanillajs": 27, "postgresql": 28, "cangjie": 29
        }
            
        self.lang_code = self.lang_code_mapping[self.lang]
        self.leetcode_headers = self.create_headers(self.leetcode_cookie, self.leetcode_crsf_token)
        
        if self.mode == "submit":
            self.client = OpenAIClient("gpt-4o", model_token=self.model_token)
        
        try:
            self.dataset = load_dataset("Elfsong/venus", self.lang, download_mode="force_redownload")
            for instance in self.dataset['train']:
                self.existing_question_ids.add(instance['question_id'])
        except ValueError as e:
            print(f"[-] The subset {self.lang} not found in the dataset: ", e)
            print(f"[-] It should be fine if you are collecting a new subset.")
                
    def create_headers(self, leetcode_cookie, leetcode_crsf_token):
        headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'cookie': leetcode_cookie,
            'origin': 'https://leetcode.com',
            'priority': 'u=1, i',
            'referer': 'https://leetcode.com/problems/two-sum/',
            'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'x-csrftoken': leetcode_crsf_token
        } 
        return headers
    
    def runtime_range(self, instance):
        instance['rt_list'] = list()
        question_id =  instance['question_id']
        question_name = instance['name']
        rt_count = len(instance['runtimeDistribution']['distribution'])
        print(f"[+] Runtime Solutions [{rt_count}]")
        
        for rt, pl in instance['runtimeDistribution']['distribution']:
            print(f"[{rt} ms]", end=" ", flush=True)
            for index in range(self.sample_num):
                time.sleep(0.3)
                response = self.runtime_retrieval(question_id=question_id, lang=self.lang, index=index, runtime=rt)
                if response and response['data']['codeWithRuntime']:
                    instance['rt_list'] += [{
                        "code": response['data']['codeWithRuntime']['code'],
                        "runtime": rt
                    }]
                    if not response['data']['codeWithRuntime']['hasNext']: break
                else:
                    break
        rt_list_len = len(instance['rt_list'])
        print(f"\n🟢 [{rt_list_len}] runtime solutions.")
        instance['rt_solution_count'] = rt_list_len
    
    def memory_range(self, instance):
        instance['mm_list'] = list()
        question_id =  instance['question_id']
        question_name = instance['name']
        mm_count = len(instance['memoryDistribution']['distribution'])
        print(f"[+] Memory Solutions [{mm_count}]")
        
        for mm, pl in instance['memoryDistribution']['distribution']:
            print(f'[{mm} kb]', end=" ", flush=True)
            for index in range(self.sample_num):
                time.sleep(0.3)
                response = self.memory_retrieval(question_id=question_id, lang=self.lang, index=index, memory=mm)
                if response and response['data']['codeWithMemory']:
                    
                    instance['mm_list'] += [{
                        "code": response['data']['codeWithMemory']['code'],
                        "memory": mm
                    }]
                    if not response['data']['codeWithMemory']['hasNext']: break
                else:
                    break
        mm_list_len = len(instance['mm_list'])
        print(f"\n🟢 [{mm_list_len}] memory solutions.")
        instance['mm_solution_count'] = mm_list_len
    
    def construct_instance(self, question):
        try:
            instance = {
                'question_id': int(question['questionId']),
                'name': question['titleSlug'],
                'content': question['content'],
                'acRate': question['acRate'],
                'difficulty': question['difficulty'],
                'topics': [topic['slug'] for topic in question['topicTags']],
            }
            
            # Code Prompts
            code_prompts = {}
            prompts = self.prompt_retrieval(question['titleSlug'])
            for code_snippet in prompts['data']['question']['codeSnippets']:
                code_prompts[code_snippet['langSlug']] = code_snippet['code']
            instance['code_prompt'] = code_prompts[self.lang]
                
            # Submission Discribution
            time.sleep(1)
            submissions = self.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code)
            if not submissions: 
                print(f"[-] Can't found any submission 🟡")
                return None
            
            time.sleep(1)
            submission_details = self.submission_detail_retrieval(submission_id=submissions[0]['id'])
            if not submission_details:
                print(f"[-] Can't retrieve the submission detail 🔴")
                return None
            
            if submission_details['runtimeDistribution']:
                instance['runtimeDistribution'] = json.loads(submission_details['runtimeDistribution'])
                self.runtime_range(instance)
                instance['runtimeDistribution'] = json.dumps(instance['runtimeDistribution'])
            else:
                print(f"[-] Can't retrieve Runtime Distribution 🔴")
                
            if submission_details['memoryDistribution']:
                instance['memoryDistribution'] = json.loads(submission_details['memoryDistribution'])
                self.memory_range(instance)
                instance['memoryDistribution'] = json.dumps(instance['memoryDistribution'])
            else:
                print(f"[-] Can't retrieve Memory Distribution 🔴")
            
            return instance
        except json.decoder.JSONDecodeError as e:
            print("[-] construct_instance: JSONDecodeError", e)
        except Exception as e:
            print("[-] construct_instance: Error", e)
            traceback.print_exc()
            return None
        
    def runtime_retrieval(self, question_id, lang, index, runtime):
        runtime_payload = json.dumps({
            "query": "\n    query codeWithRuntime($questionId: Int!, $lang: String!, $runtime: Int!, $skip: Int!) {\n  codeWithRuntime(\n    questionId: $questionId\n    lang: $lang\n    runtime: $runtime\n    skip: $skip\n  ) {\n    code\n    hasPrevious\n    hasNext\n  }\n}\n    ",
            "variables": {
                "questionId": question_id,
                "lang": lang,
                "skip": index,
                "runtime": runtime
            }
        })
        
        response = self.retrieval(runtime_payload)
        return response

    def memory_retrieval(self, question_id, lang, index, memory):
        memory_payload = json.dumps({
            "query": """query codeWithMemory($questionId: Int!, $lang: String!, $memory: Int!, $skip: Int!) {codeWithMemory(questionId: $questionId\nlang: $lang\nmemory: $memory\nskip: $skip) {code\nhasPrevious\nhasNext\n}}""",
            "variables": {
                "questionId": question_id,
                "lang": lang,
                "skip": index,
                "memory": memory
            }
        })
        
        response = self.retrieval(memory_payload)
        return response
    
    def question_retrieval(self, start=0, range_=5000):
        question_payload = json.dumps({
            "query": "query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {problemsetQuestionList: questionList(categorySlug: $categorySlug\nlimit: $limit\nskip: $skip\nfilters: $filters\n) {total: totalNum\nquestions: data {frontendQuestionId: questionFrontendId\nquestionId\nacRate\ncontent\ncodeSnippets {lang\nlangSlug\ncode}\ndifficulty\nfreqBar\nisFavor\npaidOnly: isPaidOnly\nstatus\ntitle\ntitleSlug\ntopicTags {name\nid\nslug}hasSolution\nhasVideoSolution}}}",
            "variables": {
                "categorySlug": "algorithms", "skip": start, "limit": range_, "filters": {}
            }
        })
        
        response = self.retrieval(question_payload)
        return response['data']['problemsetQuestionList']['questions']
    
    @vital_retry
    def prompt_retrieval(self, titleSlug):
        prompt_payload = json.dumps({
            "query": "query questionEditorData($titleSlug: String!) {question(titleSlug: $titleSlug) {questionId\nquestionFrontendId\ncodeSnippets {lang\nlangSlug\ncode\n}}}",
            "variables": {"titleSlug": titleSlug}
        })
        response = self.retrieval(prompt_payload)
        return response

    @vital_retry
    def submission_retrieval(self, questionSlug, lang):
        submission_payload = json.dumps({
            "query": "query submissionList($offset: Int!, $limit: Int!, $lastKey: String, $questionSlug: String!, $lang: Int, $status: Int) {questionSubmissionList(offset: $offset\nlimit: $limit\nlastKey: $lastKey\nquestionSlug: $questionSlug\nlang: $lang\nstatus: $status\n) {submissions {id\ntitleSlug\nstatus\nstatusDisplay\nruntime\nmemory\n}}}",
            "variables": {"questionSlug": questionSlug, "status": 10, "lang": lang, "offset": 0, "limit": 1, "lastKey": None}
        })
        response_json = self.retrieval(submission_payload)
        if not response_json['data']['questionSubmissionList']:
            # return None
            raise LookupError("Null Response")
        return response_json['data']['questionSubmissionList']['submissions']
    
    @vital_retry
    def submission_detail_retrieval(self, submission_id):
        submission_detail_payload = json.dumps({
            "query": "query submissionDetails($submissionId: Int!) {submissionDetails(submissionId: $submissionId) {runtime\nruntimeDistribution\nmemory\nmemoryDistribution\ncode\n}}",
            "variables": {"submissionId": submission_id}
        })
        response_json = self.retrieval(submission_detail_payload)
        if not response_json['data']['submissionDetails']:
            # return None
            raise LookupError("Null Response")
        return response_json['data']['submissionDetails']
        
    @retry
    def retrieval(self, payload):
        response = requests.request("GET", self.url, headers=self.leetcode_headers, data=payload)
        response_json = response.json()
        if not response_json['data'][next(iter(response_json['data']))]:
            raise LookupError("Null Response")
        return response_json
    
    def get_subsets(self):
        configs = get_dataset_config_names("Elfsong/venus_temp")
        subsets = [config for config in configs if config.startswith(self.lang)]
        return subsets

    def code_generation(self, instance):
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
    
    def code_submit(self, instance, code):
        question_name = instance['titleSlug']
        url = f"https://leetcode.com/problems/{question_name}/submit/"

        payload = json.dumps({
            "lang": self.lang,
            "question_id": instance['questionId'],
            "typed_code": code
        })

        response = requests.request("POST", url, headers=self.leetcode_headers, data=payload, timeout=5)
        return response.status_code
    
    def submit_pipeline(self, start, range_):
        instance_count = 0
        question_list = self.question_retrieval(start, range_)

        for question in question_list:           
            print(f"====================== [{self.lang}] Question:", question['frontendQuestionId'], question['questionId'], "https://leetcode.com/problems/"+question['titleSlug'])
            question_id = int(question['questionId'])
            if question['paidOnly']: 
                print(f"[-] Found [{question_id}] is a paid-only question, skipped ⏭️")
                continue
            if question_id in self.existing_question_ids: 
                print(f"[+] Found [{question_id}] in [Venus] datasets, skipped 😃")
                continue
            if self.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code):
                print(f"[+] Found [{question_id}] has solutions, skipped 😃")
                continue
            else:
                if self.mode == "submit":
                    print(f"[+] Submit Mode 🚀")
                    code = self.code_generation(question)
                    for _ in range(3):
                        status = self.code_submit(question, code)
                        if status == 200:
                            print("[+] Success Submission 🟢")
                            time.sleep(10)
                            break
                        time.sleep(5)
                        print("[-] Retrying 🔴")
                elif self.mode == "statistic":
                    print(f"[+] Statistic mode 🔍")
                    instance_count += 1
        return instance_count
                    
    def retrieval_pipeline(self, start, range_, sample_num):
        instances  = list()
        instance_count = 0
        self.sample_num = sample_num
        question_list = self.question_retrieval(start, range_)
        
        for index, question in enumerate(question_list):
            question_id = int(question['questionId'])
            print(f"====================== {self.lang} Question:", question['frontendQuestionId'], question['questionId'], question['titleSlug'], f"[{index+1}/{range_}]")
            if question['paidOnly']: 
                print(f"[-] Found [{question_id}] paid-only question, skipped ⏭️")
                continue
            instance_count += 1
            if question_id in self.existing_question_ids: 
                print(f"[+] Found [{question_id}] in [Venus] datasets, skipped 😃")
                continue
            else:
                print(f"[+] [{question_id}] Retrieval Mode 🚀")
                instance = self.construct_instance(question)
                if instance:
                    instances += [instance]
        
        if instances:
            print(f"====================== Uploading {len(instances)} instances to HF 🎉")
            ds = Dataset.from_pandas(pd.DataFrame(data=instances))
            ds_name = str(uuid.uuid1())
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ds.push_to_hub("Elfsong/venus_temp", f"{self.lang}-{ds_name}")
                    print("Dataset successfully pushed to hub 🎉")
                    break
                except Exception as e:
                    print(f"Failed to push dataset to hub (Attempt {attempt + 1}/{max_retries}): {e} 😕")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        print("Max retries reached. Could not push dataset to hub, skipped 😣")
        
        return instance_count
   
    def merge_pipeline(self):
        instances = list()
        instance_ids = set()
        subsets = self.get_subsets()

        print(f"🟢 Loading instances from the [Elssong/Venus] [{self.lang}] dataset...")
        try:
            ds = load_dataset("Elfsong/venus", self.lang)
            for instance in ds['train'].to_list():
                question_id = int(instance['question_id'])
                if question_id not in instance_ids:
                    instance_ids.add(instance['question_id'])
                    instances.append(instance)
            old_instance_count = len(instance_ids)
            print(f"[+] {old_instance_count} instances Loaded.")
            print("========" * 5)
        except ValueError as e:
            old_instance_count = 0
            print(f"[-] Empty dataset {args.language}, will create a new dataset.")
        
        print(f"🟢 Loading new instances from [Elfsong/venus_temp] [{self.lang}]...")
        for subset_name in tqdm(subsets):
            print(f"[+] Current Subset [{subset_name}]")    
            try: 
                if f"{args.language}-" in subset_name:
                    ds = load_dataset("Elfsong/venus_temp", subset_name)
                    for instance in ds['train'].to_list():
                        if instance['question_id'] not in instance_ids:
                            instance_ids.add(instance['question_id'])
                            instances.append(instance)
            except datasets.exceptions.DatasetGenerationError as e:
                print(f"[-] Empty Dataset {subset_name}")
            except Exception as e:
                print(f"[-] {subset_name} Error: {e}")
        new_instance_count = len(instance_ids)
        print(f"[+] {new_instance_count} instances loaded. [{new_instance_count-old_instance_count}] new instances added 🎉")
        print("========" * 5)
            
        print(f"🟢 Uploading the new [{args.language}] dataset...")
        df = pd.DataFrame(data=instances)
        df['question_id'] = df['question_id'].astype('int64')
        ds = Dataset.from_pandas(df)
        ds.push_to_hub("Elfsong/venus", args.language)
        print(f"[+] {args.language} dataset uploaded to the hub.")
        print("========" * 5)

        print("🟢 Checking the new dataset...")
        ds = load_dataset("Elfsong/venus", args.language)
        print(f"[+] {len(ds['train'])} instances in the new dataset.")
        print("========" * 5)
           
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default="golang")
    parser.add_argument("--mode", default="submit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--sample_num", type=int, default=2)
    args = parser.parse_args()

    leetcode_client = LeetCodeOperation(lang=args.language, mode=args.mode)
    instance_count = 0
    for i in tqdm(range(args.start, args.end)):
        if args.mode in ["submit", "statistic"]:
            instance_count += leetcode_client.submit_pipeline(i*args.batch, args.batch)
        elif args.mode == "retrieval": 
            instance_count += leetcode_client.retrieval_pipeline(i*args.batch, args.batch, sample_num=args.sample_num)
        elif args.mode == "merge":
            leetcode_client.merge_pipeline()
            break
        else:
            print(f"Unknown Mode: {args.mode}")
    
    print(instance_count)

