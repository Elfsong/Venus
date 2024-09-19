# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 14

# 我想让你见识一下什么是真正的勇敢，而不要错误地认为一个人手握枪支就是勇敢。
# 勇敢是：当你还未开始就已知道自己会输，可你依然要去做，而且无论如何都要把它坚持到底。
# 你很少能赢，但有时也会。

import os
import time
import json
import uuid
import requests
import argparse
import pandas as pd
from tqdm import tqdm
import src.prompts as prompts
from multiprocessing import Pool, Manager
from datasets import Dataset, load_dataset
from src.utils import OpenAIClient, DeepSeekClient

def retry(func):
    def wrap(*args, **kwargs):
        for i in range(4):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                sleep_time = 1.5**(i)
                print("🟡", end=" ", flush=True)
                time.sleep(sleep_time)
        print("🟠", end=" ", flush=True)
        return None
    return wrap

def vital_retry(func):
    def wrap(*args, **kwargs):
        for i in range(3):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print("🔴")
                time.sleep(2**(i+1))
        print("❌")
        return None
    return wrap

class LeetCodeRetrival:
    def __init__(self, lang, mode) -> None:
        self.lang = lang
        self.mode = mode
        self.instances = list()
        self.url = "https://leetcode.com/graphql/"
        self.model_token = os.getenv("CLIENT_API_KEY")
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE")
        self.leetcode_crsf_token = os.getenv("LEETCODE_CRSF_TOKEN")
        self.lang_code_mapping = { "cpp": 0, "python": 2, "golang": 10, "python3": 11, }
        self.lang_code = self.lang_code_mapping[self.lang]
        self.question_ids = set()
        
        self.leetcode_headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'cookie': self.leetcode_cookie,
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
            'x-csrftoken': self.leetcode_crsf_token
        }        
        
        if self.mode == "submit":
            self.client = OpenAIClient("gpt-4o", model_token=self.model_token)
            # self.client = DeepSeekClient("deepseek-chat", model_token=self.model_token)
        
        if self.mode == "retrieval":
            try:
                self.dataset = load_dataset("Elfsong/venus", self.lang)
                for instance in self.dataset['train']:
                    self.question_ids.add(instance['question_id'])
            except ValueError as e:
                print("Empty Language: ", e)
    
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
                
            # Submission Discribution
            time.sleep(1)
            submissions = self.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code)
            if not submissions: return None
            
            time.sleep(1)
            submission_details = self.submission_detail_retrieval(submission_id=submissions[0]['id'])
            if not submission_details: return None
            
            instance['runtimeDistribution'] = json.loads(submission_details['runtimeDistribution'])
            instance['memoryDistribution'] = json.loads(submission_details['memoryDistribution'])
            
            self.runtime_range(instance)
            self.memory_range(instance)
            
            instance['runtimeDistribution'] = json.dumps(instance['runtimeDistribution'])
            instance['memoryDistribution'] = json.dumps(instance['memoryDistribution'])
            
            return instance
        except Exception as e:
            print("[-] construct_instance error", e)
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
            print(f"====================== {self.lang} Question:", question['frontendQuestionId'], question['questionId'], question['titleSlug'])
            question_id = question['questionId']
            if question['paidOnly']: 
                print(f"[-] Found [{question_id}] is a paid-only question, skipped ⏭️")
                continue
            if question_id in self.question_ids: 
                print(f"[+] Found [{question_id}] in existing datasets, skipped 😃")
                continue
            if self.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code):
                print(f"[+] Found [{question_id}] has solutions, skipped 😃")
                continue
            else:
                if self.mode == "submit":
                    print(f"[+] Submit Mode 🚀")
                    code = self.code_generation(question)
                    for _ in range(2):
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
            if question_id in self.question_ids: 
                print(f"[+] Found [{question_id}] in existing datasets, skipped 😃")
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
            ds.push_to_hub("Elfsong/venus_temp", f"{self.lang}-{ds_name}")
        
        return instance_count
           
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default="python3") 
    parser.add_argument("--mode", default="submit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    args = parser.parse_args()

    leetcode_client = LeetCodeRetrival(lang=args.language, mode=args.mode)
    instance_count = 0
    for i in tqdm(range(args.start, args.end)):
        if args.mode in ["submit", "statistic"]:
            instance_count += leetcode_client.submit_pipeline(i*10, 10)
        elif args.mode == "retrieval": 
            instance_count += leetcode_client.retrieval_pipeline(i*10, 10, sample_num=2)
        else:
            print(f"Unknown Mode: {args.mode}")
    
    print(instance_count)

