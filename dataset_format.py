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
leetcode_cookie = os.getenv("LEETCODE_COOKIE")
leetcode_crsf_token = os.getenv("LEETCODE_CRSF_TOKEN")        
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