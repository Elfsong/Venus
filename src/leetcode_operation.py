# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 14

# 我想让你见识一下什么是真正的勇敢，而不要错误地认为一个人手握枪支就是勇敢。
# 勇敢是：当你还未开始就已知道自己会输，可你依然要去做，而且无论如何都要把它坚持到底。
# 你很少能赢，但有时也会。

import os
import time
import json
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
                sleep_time = 2**(i)
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
                sleep_time = 2**(i+2)
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
        
        if self.mode == "submit":
            self.openai_client = OpenAIClient("gpt-4o", model_token=self.model_token)
    
    def runtime_range(self, instance):
        instance['rt_list'] = list()
        for rt, pl in instance['runtimeDistribution']['distribution']:
            question_id =  instance['question_id']
            question_name = instance['name']
            print(f"Question [{question_id}] - [{question_name}] - Runtime [{rt}]")
            for index in range(self.sample_num):
                response = leetcode_client.runtime_retrieval(question_id=question_id, lang=self.lang, index=index, runtime=rt)
                if response and response['data']['codeWithRuntime']:
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
    
    def question_retrieval(self, start=0, range=5000):
        question_payload = json.dumps({
            "query": "query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {problemsetQuestionList: questionList(categorySlug: $categorySlug\nlimit: $limit\nskip: $skip\nfilters: $filters\n) {total: totalNum\nquestions: data {frontendQuestionId: questionFrontendId\nquestionId\nacRate\ncontent\ncodeSnippets {lang\nlangSlug\ncode}\ndifficulty\nfreqBar\nisFavor\npaidOnly: isPaidOnly\nstatus\ntitle\ntitleSlug\ntopicTags {name\nid\nslug}hasSolution\nhasVideoSolution}}}",
            "variables": {
                "categorySlug": "", "skip": start, "limit": range, "filters": {}
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
            "variables": {"questionSlug": questionSlug, "status": 10, "lang": lang, "offset": 0, "limit": 1 }
        })
        response_json = self.retrieval(submission_payload)
        if not response_json['data']['questionSubmissionList']:
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
        print(response.text)
    
    def submit_pipeline(self, start, range):
        question_list = self.question_retrieval(start, range)

        for question in tqdm(question_list):
            if question['paidOnly']: continue 
            if not question['hasSolution']: continue           
            if 'database' in [topic['slug'] for topic in question['topicTags']]: continue
            
            submissions = leetcode_client.submission_retrieval(questionSlug=question['titleSlug'], lang=self.lang_code)
            if submissions:
                time.sleep(1)
            else:
                print(f"{self.lang} Code Submission:", question['frontendQuestionId'], question['questionId'], question['titleSlug'])
                code = self.code_generation(question)
                self.code_submit(question, code)
                time.sleep(10)
    
    def retrieval_pipeline(self, start, range, sample_num):
        instances  = list()
        self.sample_num = sample_num
        question_list = self.question_retrieval(start, range)
        
        for question in tqdm(question_list, desc="question"):
            if not question['hasSolution']: continue  
            if question['paidOnly']: continue
            instance = self.construct_instance(question)
            if instance:
                instances += [instance]
        
        ds = Dataset.from_pandas(pd.DataFrame(data=instances))
        ds.push_to_hub("Elfsong/venus_new", f"{self.lang}-{start}-{start+range-1}")
           
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language') 
    parser.add_argument("--mode", default="retrieval")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=350)
    args = parser.parse_args()

    if args.language == "python3":
        headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'cookie': 'gr_user_id=9c46f1ce-3d85-4e02-b014-0596877e0ee8; _gid=GA1.2.706162636.1726300707; __stripe_mid=78bc9104-0d95-4914-a220-77ef87d76ef840294a; ip_check=(false, "137.132.27.180"); __cf_bm=CvXRBriTITnTS2KxkJpNe1jqKaAYwqt3VpChUneeNGc-1726470386-1.0.1.1-DebN6cjzFeZiP1vdvw8.zfVpG7gc.f.aJKdL8zTV1KZ43VloylN25kka8sscK_RXeTuTrEj__3TeTEDW64w2EQ; cf_clearance=Wl9f_nVkSUJPaBgKL8Ki4_F.99h_PpXL7saGOkfDA6U-1726470420-1.2.1.1-hZktmEbY1rE2YmDAyA.mUYqn1WQAuN3h6uKEiwaRTB2auABS9b0Ci79Hcn4ZpyAkDWeGpJUR4eKBdt2WUpGqRygQ_N8Fz2KEwXCoUwivEiU.jJQ0uUCuDFC6n7QT5JkgHHWInr_1nkSNLmalGJZT.rZHpxdIOdT9YArBYOqZzsn.jhKVqLH6bnNENZLpOMtEOxAGcBt4adxZfIxJI9gNvCV4SfQKL6nolh7V5E7KVAGiJJCtVQbFtlvEnaf0B.QYQS5Bs32tdhpsXGRNmrhM.CAfO0wbTEPOkJPTgfLtRUcZEcBBpjVeJ1zic9ZG3Olp0j39KsgDVp5S2PCzJtYUisXyNz75hkb7MHe.hKWwxF0ciUItKo85ZBg037_cK3B7mxoULvQIoJNJXRkqalBEFI0iCp9UKGpQBdvHUFQyMIlBc6cfUv_6UNVqp1R9cuHH; csrftoken=w9QK8d13qxLeVG855ILnwMp0c6oW4WGlSbLi5r5MUWryY9a6Dg1NPx68vLRgG9JU; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVKM5Mz0tNUcgvLdFTitXBpTy4NDkZKJJWmpNTCdOSmaeQWKzgmpMGVJ8O1BwLAObhJPo:1sq5p7:0hnVg7tFSlpoCXm7MqQPT5sUoTP_Uf3ZK2sN5a6Wb28; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTY1NTgwIiwiX2F1dGhfdXNlcl9iYWNrZW5kIjoiZGphbmdvLmNvbnRyaWIuYXV0aC5iYWNrZW5kcy5Nb2RlbEJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiIzNmJmM2I4N2YwNTIyMTMyN2Q2NTM5NWQ1OWQ2MjJkZDhkMDgzM2JlYmJlZWZhMGE4YjA5NzczZmRkMDUzZGY4IiwiaWQiOjE2NTU4MCwiZW1haWwiOiJkdW1pbmd6aGVAMTI2LmNvbSIsInVzZXJuYW1lIjoiRWxmc29uZyIsInVzZXJfc2x1ZyI6IkVsZnNvbmciLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvZWxmc29uZy9hdmF0YXJfMTU2NzgzMjQ2NS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3MjY0NzA0MjEsImlwIjoiMTAzLjYuMTUwLjE3MiIsImlkZW50aXR5IjoiZmUwNjczZjJhNDhkMDQ3YjkxMmIyN2UyYTBjMDJmOWYiLCJkZXZpY2Vfd2l0aF9pcCI6WyI1MjIzYmQ0NmY1MmVkYjkwNGZmZDEzYjUxMjQ2YzJmNyIsIjEwMy42LjE1MC4xNzIiXSwic2Vzc2lvbl9pZCI6NzI0OTY5MTAsIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.HTxz4lMQytdRF5EgZIPHdNbH1TrnkgJo5QwMwUsWuDs; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=1c9d800c-f6ae-4ac2-a856-5416fa3d1a96; 87b5a3c3f1a55520_gr_last_sent_cs1=Elfsong; 87b5a3c3f1a55520_gr_session_id=1c9d800c-f6ae-4ac2-a856-5416fa3d1a96; 87b5a3c3f1a55520_gr_session_id_sent_vst=1c9d800c-f6ae-4ac2-a856-5416fa3d1a96; _dd_s=rum=0&expire=1726471325358; INGRESSCOOKIE=aa0b2af0efac5af7d3773c4aef1eda15|8e0876c7c1464cc0ac96bc2edceabd27; __stripe_sid=cffd02ab-b497-44ba-b6ab-75373f17906de9ead7; 87b5a3c3f1a55520_gr_cs1=Elfsong; _ga=GA1.1.111667137.1725179955; _ga_CDRWKZTDEX=GS1.1.1726470387.18.1.1726470446.1.0.0; __cf_bm=oZoqXltU.8kP66OaMxyl2ggs6UCgUSKxqfUfki4_7fs-1726470395-1.0.1.1-yiM905PeEMfCTJ4yXXki4oM2s416_Z7kAneu7hPDPtcXb22atHcdZVtx3EL2xUh6wmEvcpIW_uW4weHHnxb86Q',
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
            'x-csrftoken': 'w9QK8d13qxLeVG855ILnwMp0c6oW4WGlSbLi5r5MUWryY9a6Dg1NPx68vLRgG9JU'
        }
        leetcode_client = LeetCodeRetrival(lang="python3", mode=args.mode, headers=headers)
        for i in tqdm(range(args.start, args.end)):
            if args.mode == "submit":
                leetcode_client.submit_pipeline(i*10, 10)
            elif args.mode == "retrieval": 
                leetcode_client.retrieval_pipeline(i*10, 10, sample_num=2)
    elif args.language == "cpp":
        headers = {
            'accept': '*/*',
            'accept-language': 'en-GB,en;q=0.9',
            'content-type': 'application/json',
            'cookie': '_gid=GA1.2.881139128.1726468155; gr_user_id=88c599e2-aa02-4bb2-b478-e228290b5b76; 87b5a3c3f1a55520_gr_session_id=eb95274d-0011-4479-9054-0d5687cb91df; ip_check=(false, "103.6.150.172"); 87b5a3c3f1a55520_gr_session_id_sent_vst=eb95274d-0011-4479-9054-0d5687cb91df; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=eb95274d-0011-4479-9054-0d5687cb91df; 87b5a3c3f1a55520_gr_last_sent_cs1=Elfsong; INGRESSCOOKIE=6e36939377a7c2dee1de98ed97778d7b|8e0876c7c1464cc0ac96bc2edceabd27; __stripe_mid=63cfa7c5-ed28-46d3-b89b-8a8ad2940c16bc7818; __stripe_sid=dde1dd55-de24-42cd-87ca-f810fbbc65c7930ecd; __cf_bm=gXJMx8ufJ6dJVpApN8nmRxQGfL0wikGtTEd0yQO4OC0-1726469430-1.0.1.1-MN3L0FsXgOVrw7rrJ1.Wh5k9as3PcwnjSSX7UFtcLJceFjjx1uNpiVGERtgAU0MTolt_JiR_LRtU8VemS1bqcw; cf_clearance=yS2.CgzwvwT0pKCBSGskqU.95uuzAYX7F_lIKV0bQkE-1726469460-1.2.1.1-BCKmV7ju7u3L1cy9tWphr7.ybISJxnylXOOMEkcc8F6Wwj0LpkXMtJSbNkdIytYorDDCmSBoS1WKjHlDMbkoOkCRhz.06vIQKUidfXO0f2CQgEIUs76JY4vItyHDB9tzxvo_2EHd5Wf4yfLwSCqt9ZkIzBSgLaL9pjZy98HCV8TySD3GdtWDxI49V6dkoN14RBXwgtXw9aYlJNNKkb1C8P6bRtrGi7Yx7qWrh2VDeJpslb9_hK2cMhPhxYFH7DLiQ5BlGfzzLUcgg0MItO1_TNouf7CcD6FZQFUmnTRjRkI7YRXmwts_OVUUMBuMMMlLuHiD6Iccnen7YenmOKGXa9v2BZIWr9KQQYMlLNT8R7AQp5EfX8.qel493IBeggkmOwu3084YC33bnMnTSDmHd74n0LENMh.RGYZ.1ahTPYJYpsjR0M3P1n1.PsJjb.eD; _dd_s=rum=0&expire=1726470365779; csrftoken=got8eSj7zU7azwzuU43vQdG1ZEn6jojtsSnMwX8ePijOZD89Kj0apMemvu6wUvkk; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRCi5NTgaKpJXm5FQqFGem56WmKGTmKSQWK7jmpAHVp-spxerg0hyZX6qQkViWCtOYX1qCTzluu3Iz89KrMlINDAyB-mMBD4k6mA:1sq5Zi:rVp0uEEgBDTCD4EU0_yJdST1wg7DaHiLoy4cCF0UzmQ; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTE2MTQ0NDUiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImJmNzg0ZDIwZDAzYjNmZGFhYWIwMzZjMTI1NWY2ZjA3ZjA5Y2M1ODMzMzUyZTJmYjliNTliNTNmNGJlZTBlZDEiLCJpZCI6MTE2MTQ0NDUsImVtYWlsIjoibWluZ3poZTAwMUBlLm50dS5lZHUuc2ciLCJ1c2VybmFtZSI6Im1pbmd6aGUwMDEiLCJ1c2VyX3NsdWciOiJtaW5nemhlMDAxIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VzZXJzL21pbmd6aGUwMDEvYXZhdGFyXzE3MjYzOTExMjgucG5nIiwicmVmcmVzaGVkX2F0IjoxNzI2NDY5NDY2LCJpcCI6IjEwMy42LjE1MC4xNzIiLCJpZGVudGl0eSI6ImZlMDY3M2YyYTQ4ZDA0N2I5MTJiMjdlMmEwYzAyZjlmIiwiZGV2aWNlX3dpdGhfaXAiOlsiNTIyM2JkNDZmNTJlZGI5MDRmZmQxM2I1MTI0NmMyZjciLCIxMDMuNi4xNTAuMTcyIl0sInNlc3Npb25faWQiOjcyNDk1NTkzLCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9.A9eUZJe-xS88ZQdUQz4mzVbvFOLscZ3jM4Sov1WUg5Y; _gat=1; 87b5a3c3f1a55520_gr_cs1=Elfsong; _ga=GA1.1.834222855.1726468155; __gads=ID=f3ee5a8ecd8f243f:T=1726468185:RT=1726469509:S=ALNI_MZ4MjvBIlxHrTDaYOQwP7JfmfHn3g; __gpi=UID=00000f0b518d97cf:T=1726468185:RT=1726469509:S=ALNI_MarGVkwL2dkg-qW2E4hJ7tLalCkyA; __eoi=ID=733ad53aa4e60c53:T=1726468185:RT=1726469509:S=AA-AfjbY0NlAm2jXEaYHV0DK2M2a; FCNEC=%5B%5B%22AKsRol92mQP8fHHb3DCRrbOt0E5s5SVrtRcRWxMXwi-VHu4ia9sZQaLRFK0CdskMhih44b9ivuCDN4rj7xcDs6DczmU7ey9KxZZJBp1eMJAFVojCLui9wq5x_uL-k0O2BPx5EmeaS6rlJNWN4V17Lo2zwwwRe4jahw%3D%3D%22%5D%5D; _ga_CDRWKZTDEX=GS1.1.1726468154.1.1.1726469518.39.0.0; __cf_bm=mDOJnGUdUE5YOhP97eaD48LKxe9kcp21PTRDlEAbCxI-1726468437-1.0.1.1-N5amPG2EUPXCiGt6zd.1kdvXneS4m.vAdAAyvvjk7bBQd31NYWDP6oL_vCFJUhWKN8CnYTh4zVYRDjqPM2jYeg',
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
            'x-csrftoken': 'got8eSj7zU7azwzuU43vQdG1ZEn6jojtsSnMwX8ePijOZD89Kj0apMemvu6wUvkk'
        }
        leetcode_client = LeetCodeRetrival(lang="cpp", mode=args.mode, headers=headers)
        for i in tqdm(range(args.start, args.end)):
            if args.mode == "submit":
                leetcode_client.submit_pipeline(i*10, 10)
            elif args.mode == "retrieval": 
                leetcode_client.retrieval_pipeline(i*10, 10, sample_num=2)
        

