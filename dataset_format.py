# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-09-23

import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset, Dataset

def retry(func):
    def wrap(*args, **kwargs):
        for i in range(3):
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

def create_headers(leetcode_cookie, leetcode_crsf_token):
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

@retry
def retrieval(payload):
    response = requests.request("GET", url, headers=leetcode_headers, data=payload)
    response_json = response.json()
    if not response_json['data'][next(iter(response_json['data']))]:
        raise LookupError("Null Response")
    return response_json
    
def prompt_retrieval(titleSlug):
    prompt_payload = json.dumps({
        "query": "query questionEditorData($titleSlug: String!) {question(titleSlug: $titleSlug) {questionId\nquestionFrontendId\ncodeSnippets {lang\nlangSlug\ncode\n}}}",
        "variables": {"titleSlug": titleSlug}
    })
    response = retrieval(prompt_payload)
    return response


parser = argparse.ArgumentParser()
parser.add_argument('--language', default="python3") 
args = parser.parse_args()

url = "https://leetcode.com/graphql/"
# leetcode_cookie = os.getenv("LEETCODE_COOKIE")
# leetcode_crsf_token = os.getenv("LEETCODE_CRSF_TOKEN")        
leetcode_cookie = "__cf_bm=i5OpyGqXW6MmEF99LLrEserWOzaAYKxUXox4tfBs2Ls-1726749698-1.0.1.1-xoa2ln9sAvxZe_Rqn2VFUBUkjVLNyivF9292dALGSQxhv1zorhbLtNJNzh19PQFgQxkIHneM_VkCsmc.vjjmmg; _gid=GA1.2.1725288676.1726749698; gr_user_id=f3298e20-88f7-45cd-9fae-3495708bb8fd; 87b5a3c3f1a55520_gr_session_id=18cad750-811b-4240-ba72-370c0fb93934; 87b5a3c3f1a55520_gr_session_id_sent_vst=18cad750-811b-4240-ba72-370c0fb93934; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=18cad750-811b-4240-ba72-370c0fb93934; 87b5a3c3f1a55520_gr_last_sent_cs1=Elfsong; INGRESSCOOKIE=aa8e9bafae60c1c30612290513894307|8e0876c7c1464cc0ac96bc2edceabd27; __stripe_mid=cc43fc80-a5bb-4161-8c8d-a996171365e6aee066; __stripe_sid=ddc34460-5188-4145-afb4-4765f4fd53112b23fd; _gat=1; cf_clearance=2m71VFyNOj5e7GcUb6Tz_7V.WL_zKQYMCPVrcE1L4PQ-1726749922-1.2.1.1-4OVtCpvKp_ZjcJb31nJY.ZQeSHXJi2QkCbMYKMA7C1FYNj5WU5nI4cRNYpgT4K1UHS8OCRFNN0SPXrfEbgVSR5gMmdqPrgYnADwGDNnZgWhhTYha29iAp.bXfE3bPfH34u_5BlIE3xZ5T0S4mrKNvIeAJdpUW0pVY1rtmv89Xed6eupbM4HevEAtQVNrS7xHBMxYXJwWuzIhOIUylzekpvPu6oebMkEy2JifFXWPkeYQ6baWzO8rr3i.HZ4QbOLfia8e7RfRu_xp0pvLd50DAGsYbU6YfwfiCAcPFikKfFXE96laZ8pOa1YTMoEpNzkNgy9GVpQE0JnzEUI9fIBuEQEdH7ygqoKx08HOlQGIIzXqchf3H73GRV0KEW55KSJbKb7aNZIqFHRL6Zh_YsD7Mybr1PEcHDm6pQouNN4JSWzHAkuJ8amKC7U70aULEzKJ; _dd_s=rum=0&expire=1726750832549; csrftoken=DqqA9MSCTuHsoQ1WiGrGwFLS0llpWZqlcZi37px62f76HTgJjEwLTfvDpV8mOkcL; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRCi5NTgaKpJXm5FQqFGem56WmKGTmKSQWK7jmpAHVp-spxerg0hyZX6qQkViWCtOYX1qCTzluu3Iz89KrMlINDAyB-mMBD4k6mA:1srGXN:bhWLMYBv_3gBDRt_p_yI3ataJuzPQX4hinkBODN9vD0; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTE2MTQ0NDUiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImJmNzg0ZDIwZDAzYjNmZGFhYWIwMzZjMTI1NWY2ZjA3ZjA5Y2M1ODMzMzUyZTJmYjliNTliNTNmNGJlZTBlZDEiLCJpZCI6MTE2MTQ0NDUsImVtYWlsIjoibWluZ3poZTAwMUBlLm50dS5lZHUuc2ciLCJ1c2VybmFtZSI6Im1pbmd6aGUwMDEiLCJ1c2VyX3NsdWciOiJtaW5nemhlMDAxIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VzZXJzL21pbmd6aGUwMDEvYXZhdGFyXzE3MjYzOTExMjgucG5nIiwicmVmcmVzaGVkX2F0IjoxNzI2NzQ5OTMzLCJpcCI6IjEzNy4xMzIuMjcuMTgwIiwiaWRlbnRpdHkiOiJmZTA2NzNmMmE0OGQwNDdiOTEyYjI3ZTJhMGMwMmY5ZiIsImRldmljZV93aXRoX2lwIjpbIjUyMjNiZDQ2ZjUyZWRiOTA0ZmZkMTNiNTEyNDZjMmY3IiwiMTM3LjEzMi4yNy4xODAiXSwic2Vzc2lvbl9pZCI6NzI4NDM5ODUsIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.P85EGBYNj787nQOE2h_FwNoWnDeSKa0wGLVLOREITEk; 87b5a3c3f1a55520_gr_cs1=Elfsong; _ga=GA1.1.166970572.1726749698; __gads=ID=298f0b60d555d1c7:T=1726749944:RT=1726749944:S=ALNI_ManBLjUUSetsC305tZq8FV0lkHQ-g; __gpi=UID=00000f10c0fced88:T=1726749944:RT=1726749944:S=ALNI_Mb4Bhds_0hM5RVEJ60rxplSmOignA; __eoi=ID=72a26375ab04053c:T=1726749944:RT=1726749944:S=AA-AfjY3F9VU6_Oy_tdobu8MLdwg; FCNEC=%5B%5B%22AKsRol8xb4h6BVeyQ4lSWvt4DWE_88IfoXlj_PzzB5VXENvHBN8oyY1uL1nukUgoeAFDi076r3T8jX1ZN7z-iD4ySxyl9kNVVVtECCjGpr9I3dNzCcOsVae91fD8-hlEtokMzjfZs52IyexCmmShQzg7ZNSkOX-iWQ%3D%3D%22%5D%5D; _ga_CDRWKZTDEX=GS1.1.1726749698.1.1.1726749950.23.0.0"
leetcode_crsf_token = "DqqA9MSCTuHsoQ1WiGrGwFLS0llpWZqlcZi37px62f76HTgJjEwLTfvDpV8mOkcL"
leetcode_headers = create_headers(leetcode_cookie, leetcode_crsf_token)

ds = load_dataset("Elfsong/venus", args.language)
dl = ds['train'].to_list()
new_dl = list()
for instance in tqdm(dl):
    name = instance['name']
    prompts = prompt_retrieval(name)
    code_prompts = {}
    for ll in prompts['data']['question']['codeSnippets']:
        code_prompts[ll['langSlug']] = ll['code']
    instance['code_prompt'] = code_prompts[args.language]
    new_dl += [instance]

ds = Dataset.from_list(new_dl)
ds.push_to_hub("Elfsong/venus_v2", args.language)