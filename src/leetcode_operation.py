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
import prompts
import requests
import argparse
import pandas as pd
from tqdm import tqdm
from utils import OpenAIClient
from multiprocessing import Pool, Manager
from datasets import Dataset, load_dataset

def retry(func):
    def wrap(*args, **kwargs):
        for i in range(3):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                sleep_time = 3**(i)
                time.sleep(sleep_time)
        return None
    return wrap

def vital_retry(func):
    def wrap(*args, **kwargs):
        for i in range(3):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                sleep_time = 2**(i)
                time.sleep(sleep_time)
        return None
    return wrap

class LeetCodeRetrival:
    def __init__(self, lang, mode, headers) -> None:
        self.lang = lang
        self.mode = mode
        self.headers = headers
        self.instances = list()
        self.url = "https://leetcode.com/graphql/"
        self.model_token = os.getenv("OPENAI_API_KEY")
        self.lang_code_mapping = { "cpp": 0, "python": 2, "golang": 10, "python3": 11, }
        self.lang_code = self.lang_code_mapping[self.lang]
        self.question_ids = set()
        
        if self.mode == "submit":
            self.openai_client = OpenAIClient("gpt-4o", model_token=self.model_token)
        
        if self.mode == "retrieval":
            self.dataset = load_dataset("Elfsong/venus", self.lang)
            for instance in self.dataset['train']:
                self.question_ids.add(instance['question_id'])
    
    def runtime_range(self, instance):
        instance['rt_list'] = list()
        for rt, pl in instance['runtimeDistribution']['distribution']:
            question_id =  instance['question_id']
            question_name = instance['name']
            print(f"Question [{question_id}] - [{question_name}] - Runtime [{rt}]")
            for index in range(self.sample_num):
                response = leetcode_client.runtime_retrieval(question_id=question_id, lang=self.lang, index=index, runtime=rt)
                if response and response['data']['codeWithRuntime']:
                    print("[+] Get A Solution 🌠")
                    instance['rt_list'] += [{
                        "code": response['data']['codeWithRuntime']['code'],
                        "runtime": rt
                    }]
                    if not response['data']['codeWithRuntime']['hasNext']: break
                else:
                    break
        instance['rt_solution_count'] = len(instance['rt_list'])
    
    def memory_range(self, instance):
        instance['mm_list'] = list()
        for mm, pl in instance['memoryDistribution']['distribution']:
            question_id =  instance['question_id']
            question_name = instance['name']
            print(f"Question [{question_id}] - [{question_name}] - Memory [{mm}]")
            for index in range(self.sample_num):
                response = leetcode_client.memory_retrieval(question_id=question_id, lang=self.lang, index=index, memory=mm)
                if response and response['data']['codeWithMemory']:
                    print("[+] Get A Solution 🌠")
                    instance['mm_list'] += [{
                        "code": response['data']['codeWithMemory']['code'],
                        "memory": mm
                    }]
                    if not response['data']['codeWithMemory']['hasNext']: break
                else:
                    break
        instance['mm_solution_count'] = len(instance['mm_list'])
    
    def construct_instance(self, question):
        try:
            instance = {
                'question_id': question['questionId'],
                'name': question['titleSlug'],
                'content': question['content'],
                'acRate': question['acRate'],
                'difficulty': question['difficulty'],
                'topics': [topic['slug'] for topic in question['topicTags']],
            }
                
            # Skip dataset questions
            if 'database' in instance['topics']: return None
            
            # Submission Discribution
            submissions = leetcode_client.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code)
            if not submissions: return None
            
            submission_details = leetcode_client.submission_detail_retrieval(submission_id=submissions[0]['id'])
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

    # @vital_retry
    def submission_retrieval(self, questionSlug, lang):
        submission_payload = json.dumps({
            "query": "query submissionList($offset: Int!, $limit: Int!, $lastKey: String, $questionSlug: String!, $lang: Int, $status: Int) {questionSubmissionList(offset: $offset\nlimit: $limit\nlastKey: $lastKey\nquestionSlug: $questionSlug\nlang: $lang\nstatus: $status\n) {submissions {id\ntitleSlug\nstatus\nstatusDisplay\nruntime\nmemory\n}}}",
            "variables": {"questionSlug": questionSlug, "status": 10, "lang": lang, "offset": 0, "limit": 1 }
        })
        response_json = self.retrieval(submission_payload)
        if not response_json['data']['questionSubmissionList']:
            raise LookupError("Null Response")
        return response_json['data']['questionSubmissionList']['submissions']
    
    # @vital_retry
    def submission_detail_retrieval(self, submission_id):
        submission_detail_payload = json.dumps({
            "query": "query submissionDetails($submissionId: Int!) {submissionDetails(submissionId: $submissionId) {runtime\nruntimeDistribution\nmemory\nmemoryDistribution\ncode\n}}",
            "variables": {"submissionId": submission_id}
        })
        response_json = self.retrieval(submission_detail_payload)
        if not response_json['data']['submissionDetails']:
            raise LookupError("Null Response")
        return response_json['data']['submissionDetails']
        
    @retry
    def retrieval(self, payload):
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
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
        response = self.openai_client.inference(messages)
        return response['solution']
    
    def code_submit(self, instance, code):
        question_name = instance['titleSlug']
        url = f"https://leetcode.com/problems/{question_name}/submit/"

        payload = json.dumps({
            "lang": self.lang,
            "question_id": instance['questionId'],
            "typed_code": code
        })
        response = requests.request("POST", url, headers=self.headers, data=payload, timeout=5)
        return response.status_code
    
    def submit_pipeline(self, start, range_):
        question_list = self.question_retrieval(start, range_)
        candidates = list()
        for question in question_list:
            if question['paidOnly']: continue        
            if 'database' in [topic['slug'] for topic in question['topicTags']]: continue
            print(f"====================== {self.lang} Question:", question['frontendQuestionId'], question['questionId'], question['titleSlug'])
            
            submissions = leetcode_client.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code)
            if submissions:
                print(f"[+] Found solution 😃")
                time.sleep(0.5)
            else:
                candidates += [question['frontendQuestionId']]
                if self.mode == "submit":
                    print(f"[+] Generating code and submitting 🚀")
                    code = self.code_generation(question)
                    for _ in range(2):
                        status = self.code_submit(question, code)
                        if status == 200:
                            print("[+] Success Submission  🟢")
                            time.sleep(10)
                            break
                        time.sleep(5)
                        print("[-] Retrying 🔴")
        return candidates
                    
    def retrieval_pipeline(self, start, range_, sample_num):
        instances  = list()
        self.sample_num = sample_num
        question_list = self.question_retrieval(start, range_)
        
        for question in tqdm(question_list, desc="question"):
            if not question['hasSolution']: continue  
            if question['paidOnly']: continue
            if question['questionId'] in self.question_ids: continue
            instance = self.construct_instance(question)
            if instance:
                instances += [instance]
        
        if instances:
            ds = Dataset.from_pandas(pd.DataFrame(data=instances))
            ds_name = str(uuid.uuid1())
            ds.push_to_hub("Elfsong/venus_temp", f"{self.lang}-{ds_name}")
           
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default="python3") 
    parser.add_argument("--mode", default="submit")
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--end", type=int, default=350)
    args = parser.parse_args()

    if args.language == "python3":
        headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'cookie': 'gr_user_id=9c46f1ce-3d85-4e02-b014-0596877e0ee8; _gid=GA1.2.706162636.1726300707; __stripe_mid=78bc9104-0d95-4914-a220-77ef87d76ef840294a; cf_clearance=1owrkD8IRsQXTDy6_H_QGgf.syN_BE6As5L2WjoVH0o-1726484459-1.2.1.1-zQCb_IwMA8aW3W3Dl0BREgYBTNj85qmC2D8fsks.jZtve9mp2CesWYDADyyDf0XRIzUeqS_oTQ2a1_QObVG.bbFZcFbI6dlSxXxZ1ANFCqu0zsvtbKAI5O29o1ZULd_Bh0pI3oP1nC8jHgLtiOA3puluGxYnNE.Lmmce2G1xm0Agbl4w9mkhPwsrTb2zPYEB3ZVD3bhpVFMGY3vWOQQrUxp88WvPJ8AXlrw3RYe6IMIVX3lq7aPxmksuRhIuN2wGrRhZ6s5_JOY3LxNShledWm1pHph9hBYNlLcDPWXm7y81JwDBGhH1ok2uDL.WlVABHNphZccX7q71CbH.nq7_x2kxJUzDZxfQepAKkV5iiyGbLhyL8PR9S3r6JxWxfm_jVlY0r8BIfZSWwH5tJb2W4RdGD64HO_jfFKLczdWI1ulOwdFCWyPr2nQRO6vIxKfu; csrftoken=ru31nv8NWJBPn0GlNk9Snx7w0J6CpIE6jfxkL7PN0e7VPthgLXBsyTdAhaumMaKl; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVKM5Mz0tNUcgvLdFTitXBpTy4NDkZKJJWmpNTCdOSmaeQWKyQm5mXXpWRamBgiE8_1axzzUkDqk8Hao4FAOnOShw:1sq9TZ:QYu_QH6ZKsuiB0qe1DZF9hoF7SC-GMimw4weHgliX5Q; 87b5a3c3f1a55520_gr_last_sent_cs1=Elfsong; INGRESSCOOKIE=13bbbea676d475ed0e9bf577169cf313|8e0876c7c1464cc0ac96bc2edceabd27; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTY1NTgwIiwiX2F1dGhfdXNlcl9iYWNrZW5kIjoiZGphbmdvLmNvbnRyaWIuYXV0aC5iYWNrZW5kcy5Nb2RlbEJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiIzNmJmM2I4N2YwNTIyMTMyN2Q2NTM5NWQ1OWQ2MjJkZDhkMDgzM2JlYmJlZWZhMGE4YjA5NzczZmRkMDUzZGY4IiwiaWQiOjE2NTU4MCwiZW1haWwiOiJkdW1pbmd6aGVAMTI2LmNvbSIsInVzZXJuYW1lIjoiRWxmc29uZyIsInVzZXJfc2x1ZyI6IkVsZnNvbmciLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvZWxmc29uZy9hdmF0YXJfMTU2NzgzMjQ2NS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3MjY1NzEwMzUsImlwIjoiMTM3LjEzMi4yNy4xODAiLCJpZGVudGl0eSI6ImZlMDY3M2YyYTQ4ZDA0N2I5MTJiMjdlMmEwYzAyZjlmIiwiZGV2aWNlX3dpdGhfaXAiOlsiNTIyM2JkNDZmNTJlZGI5MDRmZmQxM2I1MTI0NmMyZjciLCIxMzcuMTMyLjI3LjE4MCJdLCJzZXNzaW9uX2lkIjo3MjUxNDI4OCwiX3Nlc3Npb25fZXhwaXJ5IjoxMjA5NjAwfQ.tpqswjIPpvxiWVKG7pZBeksB-AxhU5YrjRNcYDyWLcA; ip_check=(false, "137.132.27.180"); 87b5a3c3f1a55520_gr_session_id=039ddac0-68c2-4e04-8dac-e115cb059540; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=039ddac0-68c2-4e04-8dac-e115cb059540; 87b5a3c3f1a55520_gr_session_id_sent_vst=039ddac0-68c2-4e04-8dac-e115cb059540; __cf_bm=ZX9wQIhwhuXHTXcXKgWn9OZmavWoELmPqyY4MIxmc1E-1726573733-1.0.1.1-Yu50ZX45tsS6hZGs.SR76Xth6inqadOd6bIyOOVMgTVYAzseBaurriCKU5DqJj0V.I3JfxhaVfkwHw.V92dvWg; _gat=1; 87b5a3c3f1a55520_gr_cs1=Elfsong; _ga=GA1.1.111667137.1725179955; _dd_s=rum=0&expire=1726574640468; _ga_CDRWKZTDEX=GS1.1.1726573733.25.1.1726573765.28.0.0',
            'origin': 'https://leetcode.com',
            'priority': 'u=1, i',
            'referer': 'https://leetcode.com/problems/number-of-subarrays-with-and-value-of-k/',
            'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'x-csrftoken': 'ru31nv8NWJBPn0GlNk9Snx7w0J6CpIE6jfxkL7PN0e7VPthgLXBsyTdAhaumMaKl'
        }        
    elif args.language == "cpp":
        headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en;q=0.9',
            'content-type': 'application/json',
            'cookie': '__cf_bm=hwPOXWMt4pazNPEn_1XOZVaRKg6Pp8vNtrsOvoL.gcU-1726555932-1.0.1.1-_B3WDzfnaiTrzcne9PNl3dnrHyMQKTZA4qsSuNYiG8VTPrSMOxvn0tfyq0aEFJ_tdNuDIB7N5n0krasPVwAB6Q; _gid=GA1.2.829174000.1726555933; _gat=1; gr_user_id=7c78d549-da82-46a3-9fe1-96401331ca77; 87b5a3c3f1a55520_gr_session_id=445e6197-8643-4f70-a20e-b15237bbad57; ip_check=(false, "137.132.27.180"); 87b5a3c3f1a55520_gr_session_id_sent_vst=445e6197-8643-4f70-a20e-b15237bbad57; cf_clearance=dgyyZSRwhAAnR8Fq9HSyIFRRN3f_nS7RpGL2CtHHDx4-1726555934-1.2.1.1-9njVWgExmFz0ljwY1.AC1DKXQULWouN63p3c515oEJE6F4y.mwi_TeM3nuhkHY6K4ci5kUuPyPmR8VeWh13H00NhQvu1jApUsxckACrcYXDWzr6iD8s_9yiAn6GWrZvcRe.eCckUBJG.pxWAcyySKBBrwnB5mJ9l4ykCA7T31ZI5tSpK8fYOIa8RlNbIjyUMknvn8VoxYmLi1sNxWWtZ1UJJd7B0sG8WIcDMzy_YgXUHxghNcSJTb3G70BG3CMa460_7AaRqMJ7zkrD3M8BlLTeZmZGUapqGoULe73.pIaGnLSjDno503kVcmIDEEZdDzs07WKe2s8h7A94SVjs.0c9ao82aL8bwBhOljX9ypJL4ybRUajEUurlibosNkxCp0nh_vJhk47fKd1id5_2dzn.aNQxo9vKS0rBlFhrnllYE7G9.Pw1vRIPtOtrzlvf8; csrftoken=pZYQ3RmWmTUYAAVQiKGZOQzqgYdxVGiiZDrCz0tAqAhOcccNsZqHtizTIdViO3Dz; messages=W1siX19qc29uX21lc3NhZ2UiLDAsMjUsIlN1Y2Nlc3NmdWxseSBzaWduZWQgaW4gYXMgbWluZ3poZTAwMS4iXV0:1sqS4T:kUAaFjXqHzHFj9EO5DnJMKdbK-j8zDMDDfl1glyJNwY; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTE2MTQ0NDUiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImJmNzg0ZDIwZDAzYjNmZGFhYWIwMzZjMTI1NWY2ZjA3ZjA5Y2M1ODMzMzUyZTJmYjliNTliNTNmNGJlZTBlZDEiLCJpZCI6MTE2MTQ0NDUsImVtYWlsIjoibWluZ3poZTAwMUBlLm50dS5lZHUuc2ciLCJ1c2VybmFtZSI6Im1pbmd6aGUwMDEiLCJ1c2VyX3NsdWciOiJtaW5nemhlMDAxIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VzZXJzL21pbmd6aGUwMDEvYXZhdGFyXzE3MjYzOTExMjgucG5nIiwicmVmcmVzaGVkX2F0IjoxNzI2NTU1OTQxLCJpcCI6IjEzNy4xMzIuMjcuMTgwIiwiaWRlbnRpdHkiOiJmZTA2NzNmMmE0OGQwNDdiOTEyYjI3ZTJhMGMwMmY5ZiIsImRldmljZV93aXRoX2lwIjpbIjUyMjNiZDQ2ZjUyZWRiOTA0ZmZkMTNiNTEyNDZjMmY3IiwiMTM3LjEzMi4yNy4xODAiXSwic2Vzc2lvbl9pZCI6NzI1OTgzNjAsIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.H5lCwPSicaUx3dgmPTbXgEhQIjoP8xTvWjxaZQnuuJs; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=445e6197-8643-4f70-a20e-b15237bbad57; 87b5a3c3f1a55520_gr_last_sent_cs1=mingzhe001; INGRESSCOOKIE=c615ca062de54d71f512ac0c3a702bde|8e0876c7c1464cc0ac96bc2edceabd27; _dd_s=rum=0&expire=1726556844885; 87b5a3c3f1a55520_gr_cs1=mingzhe001; _ga=GA1.1.695003655.1726555933; _ga_CDRWKZTDEX=GS1.1.1726555932.1.1.1726555949.43.0.0',
            'origin': 'https://leetcode.com',
            'priority': 'u=1, i',
            'referer': 'https://leetcode.com/problems/',
            'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'x-csrftoken': 'pZYQ3RmWmTUYAAVQiKGZOQzqgYdxVGiiZDrCz0tAqAhOcccNsZqHtizTIdViO3Dz'
        }
        
    leetcode_client = LeetCodeRetrival(lang=args.language, mode=args.mode, headers=headers)
    
    q_candidates = list()
    for i in tqdm(range(args.start, args.end)):
        if args.mode in ["submit", "statistic"]:
            candidates = leetcode_client.submit_pipeline(i*10, 10)
            q_candidates += [candidates]
        elif args.mode == "retrieval": 
            leetcode_client.retrieval_pipeline(i*10, 10, sample_num=2)
            
            
    print(q_candidates)
    print(len(q_candidates))
        

