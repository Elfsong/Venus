
# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024 / 09 / 14

# 我想让你见识一下什么是真正的勇敢，而不要错误地认为一个人手握枪支就是勇敢。
# 勇敢是：当你还未开始就已知道自己会输，可你依然要去做，而且无论如何都要把它坚持到底。
# 你很少能赢，但有时也会。

import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager
from datasets import Dataset, load_dataset


class LeetCodeRetrival:
    def __init__(self, lang, headers) -> None:
        self.lang = lang
        self.instances = list()
        self.url = "https://leetcode.com/graphql/"
        self.headers = headers
        
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
            "query": "query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {problemsetQuestionList: questionList(categorySlug: $categorySlug\nlimit: $limit\nskip: $skip\nfilters: $filters\n) {total: totalNum\nquestions: data {acRate\ncontent\ncodeSnippets {lang\nlangSlug\ncode}\ndifficulty\nfreqBar\nfrontendQuestionId: questionFrontendId\nisFavor\npaidOnly: isPaidOnly\nstatus\ntitle\ntitleSlug\ntopicTags {name\nid\nslug}hasSolution\nhasVideoSolution}}}",
            "variables": {
                "categorySlug": "", "skip": start, "limit": range, "filters": {}
            }
        })
        
        response = self.retrieval(question_payload)
        return response['data']['problemsetQuestionList']['questions']
    
    def submission_retrieval(self, questionSlug, lang):
        submission_payload = json.dumps({
            "query": "query submissionList($offset: Int!, $limit: Int!, $lastKey: String, $questionSlug: String!, $lang: Int, $status: Int) {questionSubmissionList(offset: $offset\nlimit: $limit\nlastKey: $lastKey\nquestionSlug: $questionSlug\nlang: $lang\nstatus: $status\n) {submissions {id\ntitleSlug\nstatus\nstatusDisplay\nruntime\nmemory\n}}}",
            "variables": {"questionSlug": questionSlug, "status": 10, "lang": lang, "offset": 0, "limit": 3, "lastKey": "null" }
        })
        response = self.retrieval(submission_payload)
        return response['data']['questionSubmissionList']['submissions']
    
    def submission_detail_retrieval(self, submission_id):
        try:
            submission_detail_payload = json.dumps({
                "query": "query submissionDetails($submissionId: Int!) {submissionDetails(submissionId: $submissionId) {runtime\nruntimeDistribution\nmemory\nmemoryDistribution\ncode\n}}",
                "variables": {"submissionId": submission_id}
            })
            response = self.retrieval(submission_detail_payload)
            return response['data']['submissionDetails']
        except Exception as e:
            print("submission_detail_retrieval error", e)
            return None

    def retrieval(self, payload):
        try:
            response = requests.request("POST", self.url, headers=self.headers, data=payload)
            return response.json()
        except Exception as e:
            print(f"Retrieval Error: {e} Response: {response}")
            return None
    
    def runtime_range(self, instance):
        instance['rt_list'] = list()
        for rt, pl in instance['runtimeDistribution']['distribution']:
            question_id =  instance['question_id']
            question_name = instance['name']
            print(f"Question [{question_id}] - [{question_name}] - Runtime [{rt}]")
            for index in range(6):
                response = leetcode_client.runtime_retrieval(question_id=question_id, lang=self.lang, index=index, runtime=rt)
                if response and response['data']['codeWithRuntime']:
                    instance['rt_list'] += [{
                        "code": response['data']['codeWithRuntime']['code'],
                        "runtime": rt
                    }]
                    if not response['data']['codeWithRuntime']['hasNext']:
                        break
                else:
                    break
        instance['rt_solution_count'] = len(instance['rt_list'])
    
    def memory_range(self, instance):
        instance['mm_list'] = list()
        for mm, pl in instance['memoryDistribution']['distribution']:
            question_id =  instance['question_id']
            question_name = instance['name']
            print(f"Question [{question_id}] - [{question_name}] - Memory [{mm}]")
            for index in range(5):
                response = leetcode_client.memory_retrieval(question_id=question_id, lang=self.lang, index=index, memory=mm)
                if response and response['data']['codeWithMemory']:
                    instance['mm_list'] += [{
                        "code": response['data']['codeWithMemory']['code'],
                        "memory": mm
                    }]
                    if not response['data']['codeWithMemory']['hasNext']:
                        break
                else:
                    break
        instance['mm_solution_count'] = len(instance['mm_list'])
    
    def construct_instance(self, question):
        try:
            instance = {
                'question_id': question['frontendQuestionId'],
                'name': question['titleSlug'],
                'content': question['content'],
                'acRate': question['acRate'],
                'difficulty': question['difficulty'],
                'topics': [topic['slug'] for topic in question['topicTags']],
            }
            print("[+] Current Question: ", instance['question_id'])
                
            # Skip dataset questions
            if 'database' in instance['topics']: return None
            
            if question['paidOnly'] == True: return None
            
            # Submission Discribution
            submissions = leetcode_client.submission_retrieval(questionSlug=question['titleSlug'], lang=2)
            if not submissions: return None
            
            submission_details = leetcode_client.submission_detail_retrieval(submission_id=submissions[0]['id'])
            if not submission_details: return None
            
            instance['runtimeDistribution'] = json.loads(submission_details['runtimeDistribution'])
            instance['memoryDistribution'] = json.loads(submission_details['memoryDistribution'])
            
            self.runtime_range(instance)
            self.memory_range(instance)
            
            instance['runtimeDistribution'] = json.dumps(instance['runtimeDistribution'])
            instance['memoryDistribution'] = json.dumps(instance['memoryDistribution'])
            
            print("Done Question: ", instance['question_id'])
            
            return instance
        except Exception as e:
            print("[-] construct_instance error", e)
            return None
        
    
    def pipeline(self, start, range):
        question_list = self.question_retrieval(start, range)

        results = list()
        with Pool(1) as pool:            
            for question in question_list:
                results.append(pool.apply_async(self.construct_instance, args=(question,)))
            self.instances = [result.get() for result in results]
        
        # for question in tqdm(question_list):
        #     self.instances += [self.construct_instance(question)]
        
        self.instances = [instance for instance in self.instances if instance]
        
        ds = Dataset.from_pandas(pd.DataFrame(data=self.instances))
        ds.push_to_hub("Elfsong/Venus", f"python3-{start}-{start+range-1}")
           
if __name__ == "__main__": 
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
        'authorization': '',
        'content-type': 'application/json',
        'cookie': 'gr_user_id=9c46f1ce-3d85-4e02-b014-0596877e0ee8; _gid=GA1.2.706162636.1726300707; __stripe_mid=78bc9104-0d95-4914-a220-77ef87d76ef840294a; INGRESSCOOKIE=aa0b2af0efac5af7d3773c4aef1eda15|8e0876c7c1464cc0ac96bc2edceabd27; ip_check=(false, "137.132.27.180"); cf_clearance=74XO8mwa8_WcMhN6V4TCcYi6j.COAv9RU5vVjejgUow-1726390814-1.2.1.1-UVZz9RdDYAF.wOuDovmjLmpDgxlpQvOlOiYC.TEF_BVJjCgcbVpUPrC0cB6cva.50yaFySAIRyS1GIM02gyxcRtdbDiUeUKmCHOoF1ngXkJGXRYJVQmGNltlUgizc1Ghuk1r0tDoy4LTX3hbQwVj3MxwY0PzOsBlDKhLltTpZpvIaDvfQkA5Wy7R0_ck202IK86qbgzv3PCZEFWj378hcByqbGTB0J3x0RREWX2BRRi2JA5n0oo_O1Nm.D9qU8jZh_.X9ZRmHPXMqPBDpNezOk5MfmIUkyw5JBMYr7cyceEJj3VGm6PM00g1_AuNd6EEd2Yzow_3FIhmfM75CflKy8qZoroiViW57oa_4dsGB0TAPMJWx2iIhusrL5rnx8cboeIBwyVyuI7.dkhSX3H8cU9I1qMD9v8fB8q8UD2pOD9weFkaBTtdnPWfvd627FkC; csrftoken=jHTqRqrvc11VHVj1HTNILdRikAN1yxNkySP95H2BEPqcTypowKT3DXum4i3eNGEu; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVSM7PS8ssyk1NUcjNzEuvykh1yCst1ktNKdUrTtdTitUhaEBxZnoeUHd-aQk-5cGlyclAkbTSnJxKmJbMPIXEYpi1BgaGdLHONScNqH6Yey0WABIvqhA:1spl7F:1BIQJLJtdK_5UjNFOueTWFaZfMyY50S9WcD-bbfBTPk; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTY1NTgwIiwiX2F1dGhfdXNlcl9iYWNrZW5kIjoiZGphbmdvLmNvbnRyaWIuYXV0aC5iYWNrZW5kcy5Nb2RlbEJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiIzNmJmM2I4N2YwNTIyMTMyN2Q2NTM5NWQ1OWQ2MjJkZDhkMDgzM2JlYmJlZWZhMGE4YjA5NzczZmRkMDUzZGY4IiwiaWQiOjE2NTU4MCwiZW1haWwiOiJkdW1pbmd6aGVAMTI2LmNvbSIsInVzZXJuYW1lIjoiRWxmc29uZyIsInVzZXJfc2x1ZyI6IkVsZnNvbmciLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvZWxmc29uZy9hdmF0YXJfMTU2NzgzMjQ2NS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3MjYzOTA4MjEsImlwIjoiMTM3LjEzMi4yNy4xODAiLCJpZGVudGl0eSI6ImZlMDY3M2YyYTQ4ZDA0N2I5MTJiMjdlMmEwYzAyZjlmIiwiZGV2aWNlX3dpdGhfaXAiOlsiNTIyM2JkNDZmNTJlZGI5MDRmZmQxM2I1MTI0NmMyZjciLCIxMzcuMTMyLjI3LjE4MCJdLCJzZXNzaW9uX2lkIjo3MjQxOTY3OSwiX3Nlc3Npb25fZXhwaXJ5IjoxMjA5NjAwfQ.C_nLXGO7urQI9s3eqDetxn6C-PpBvS5KeyKC9JZZFpk; 87b5a3c3f1a55520_gr_last_sent_cs1=Elfsong; 87b5a3c3f1a55520_gr_session_id=920de2ce-08aa-49bb-9a00-c1257671fc95; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=920de2ce-08aa-49bb-9a00-c1257671fc95; 87b5a3c3f1a55520_gr_session_id_sent_vst=920de2ce-08aa-49bb-9a00-c1257671fc95; __stripe_sid=6e9c06af-96e0-4e31-8690-f8097acfca2c8fd184; _dd_s=rum=0&expire=1726414887586; _gat=1; 87b5a3c3f1a55520_gr_cs1=Elfsong; _ga=GA1.1.111667137.1725179955; __cf_bm=ncb15VMxZS_HpXEgEHmvXXhEWtcCzErpREiK8YArjFo-1726414591-1.0.1.1-V4awEkQ0kry5XPpCMsXY.zy1OgWdxGMJ28qobo5dvRp6Rm8VKGJZSPc5sY_.ISXevQyY2UmZrxUTABHievqvuQ; _ga_CDRWKZTDEX=GS1.1.1726413685.13.1.1726414592.25.0.0; csrftoken=z4sGm2yxyOt8uw9SDZyoKT8fX1cefDXB0CPXrMZiN0SW1kAEpmIAth7MlRir2UVc',
        'origin': 'https://leetcode.com',
        'priority': 'u=1, i',
        'random-uuid': 'f90d35df-e7c0-d272-1ee7-27e90891caa6',
        'referer': 'https://leetcode.com/problems/two-sum/',
        'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'uuuserid': '5223bd46f52edb904ffd13b51246c2f7',
        'x-csrftoken': 'jHTqRqrvc11VHVj1HTNILdRikAN1yxNkySP95H2BEPqcTypowKT3DXum4i3eNGEu'
    }
    leetcode_client = LeetCodeRetrival(lang="python3", headers=headers)
    leetcode_client.pipeline(0, 100)