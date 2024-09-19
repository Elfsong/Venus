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
        headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'cookie': '_gid=GA1.2.1535148603.1726735303; gr_user_id=32512bd3-c65f-458c-8681-f8f5463045b1; ip_check=(false, "137.132.27.180"); __stripe_mid=6126618b-e583-4cb1-b5bb-a8c5310062e2ede87a; __stripe_sid=52c6eae5-ae84-4441-9aeb-d8700b5f4d21a79274; __cf_bm=7VZawBsCFAfn.8DMAt8h16WS1xaNHAg7zuO9GGkIqxI-1726736317-1.0.1.1-hBvJs5cVIPWLgzP1n_0vJRvCAdtRJs.jxzWiP1z_X7_PiI0mtTAmnXxFJQxRNknVCDEuOxINMVT0Gee1OC688Q; cf_clearance=l4xrgNsX4dTH94HAKVVmxw2gpFvcVmnFuMRCcUiXh7I-1726736324-1.2.1.1-xHvFvIDu9KEMXSOsHBnC80NzvKquIaaxfETYBVuoaUJlMZt6sOYgJU3RBdjpw3Vqv6lHTQFV9sdAJcRcwfpP4UUPrZOgZv2_45Qv2dep4tIFqwa8ao_nbbPeNZJgU9SaEoCY2UN4dqnjYt9XO818_Tbko3yAM2mPRVZuSquf9P12XpwFle24u.pb841mV0axFAIq4MUS6QPaJ5yGPhU7lxfhmyoXs8HYaeUMiOeCmxndvgHXC0I03WNamUc8MvgSrbsydIRFTVw1ERmQHboGxv5CxE9nu.36hUZ9._JacKRvuhldU8tFBr4QFnYT.Rab2y41C_YBGY4NVugKS5jBvJOf8ZrNuyml1PJFRXxlZuqzkfFIrNJOk_GcyWUAp.jglntlwL9RXRyUuc2gBGQ4OzwPUrzUk_onanCnRP2jnfmh98O_ucajDPyESlH5rZLN; csrftoken=REahq1iFS4OlpnLrEnuScC6AnjjJO2GNgEi5KPdmtOzqFbFFi2evGTUqCTUPgH3z; messages=.eJx9jUEKwjAQAL-y5JKDqWxCVeg3PEkpIcbYrrSJdJOKvt4ieBOvwwzTtsLaG6dop8Ds-iAUKoNKnFKZgcuZ_Uz3TCkCZ5cLAzHE9ADpfKYlSCgx0wjSoKkrjZXeg8amNo05bBAbRCk69XOy-0xgcEsApj6GC6SSt3_0Y_F-Jdcyjs9vQhEcw0Sxfw0BUa999waohESM:1srD05:JKpPDPPwkc1KIOcmZkxrPKkk2gLpgxqQAA-iNbMTgrE; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTE2MTQ0NDUiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImJmNzg0ZDIwZDAzYjNmZGFhYWIwMzZjMTI1NWY2ZjA3ZjA5Y2M1ODMzMzUyZTJmYjliNTliNTNmNGJlZTBlZDEiLCJpZCI6MTE2MTQ0NDUsImVtYWlsIjoibWluZ3poZTAwMUBlLm50dS5lZHUuc2ciLCJ1c2VybmFtZSI6Im1pbmd6aGUwMDEiLCJ1c2VyX3NsdWciOiJtaW5nemhlMDAxIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VzZXJzL21pbmd6aGUwMDEvYXZhdGFyXzE3MjYzOTExMjgucG5nIiwicmVmcmVzaGVkX2F0IjoxNzI2NzM2MzM3LCJpcCI6IjEzNy4xMzIuMjcuMTgwIiwiaWRlbnRpdHkiOiJmZTA2NzNmMmE0OGQwNDdiOTEyYjI3ZTJhMGMwMmY5ZiIsImRldmljZV93aXRoX2lwIjpbIjUyMjNiZDQ2ZjUyZWRiOTA0ZmZkMTNiNTEyNDZjMmY3IiwiMTM3LjEzMi4yNy4xODAiXSwic2Vzc2lvbl9pZCI6NzI4MjYzNTcsIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.x3hdolKddT6P8ryGYo-_-yOzfwo6dXVYWLe3ejDJ2hY; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=594c25ce-1ca5-41bc-9b10-21520d4c781c; 87b5a3c3f1a55520_gr_last_sent_cs1=mingzhe001; 87b5a3c3f1a55520_gr_session_id=594c25ce-1ca5-41bc-9b10-21520d4c781c; 87b5a3c3f1a55520_gr_session_id_sent_vst=594c25ce-1ca5-41bc-9b10-21520d4c781c; _dd_s=rum=0&expire=1726737257599; INGRESSCOOKIE=2d72a96471cb6b2879d9ed56c7a65fb6|8e0876c7c1464cc0ac96bc2edceabd27; 87b5a3c3f1a55520_gr_cs1=mingzhe001; _ga=GA1.1.896039270.1726735303; __gads=ID=cc473ccc7cc06262:T=1726736368:RT=1726736368:S=ALNI_MZJo6sZU24BDi16AVPF28IAfTO9Ng; __gpi=UID=00000f10a7b71202:T=1726736368:RT=1726736368:S=ALNI_Maa6mAh4zcDBGnXjNmwvpfyY-ilAQ; __eoi=ID=8455bdac700a73de:T=1726736368:RT=1726736368:S=AA-AfjaHbZfXSVgXVR1VIF90W5A3; FCNEC=%5B%5B%22AKsRol953ghGkNfTnoUXRc66uUQxB_5i-x5RlOWTP0K4iypjCO2Fag9QYHE0Jyoy_YIbLfGWNEeoWoi4RN5brpxs83Bfybr1R1mx1JopUbHW9Nf4wemkifGHld_hjDLnHqlx6ERZB68p3PKXcJ8965SuuA41jTPFxw%3D%3D%22%5D%5D; _ga_CDRWKZTDEX=GS1.1.1726735302.1.1.1726736374.3.0.0',
            'origin': 'https://leetcode.com',
            'priority': 'u=1, i',
            'referer': 'https://leetcode.com/problems/add-two-numbers/',
            'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'x-csrftoken': 'REahq1iFS4OlpnLrEnuScC6AnjjJO2GNgEi5KPdmtOzqFbFFi2evGTUqCTUPgH3z'
        }
        response = requests.request("POST", url, headers=headers, data=payload, timeout=5)
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

