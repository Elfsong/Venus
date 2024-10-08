# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/08/31

import time
import json
import hashlib
from openai import OpenAI

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

def generate_hash(input_string, algorithm='sha256'):
    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        print(f"Error: Unsupported hashing algorithm '{algorithm}'")
        return None

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

class Client(object):
    def __init__(self, model_name, model_token) -> None:
        self.model_name = model_name
        self.model_token = model_token
        self.client = None
    
    def inference(self, context):
        raise NotImplementedError("Don't call the base class directly.")

class DeepSeekClient(Client):
    def __init__(self, model_name, model_token) -> None:
        super().__init__(model_name, model_token)
        self.client = OpenAI(base_url="https://api.deepseek.com", api_key=self.model_token)
    
    def inference(self, messages):
        response = self.client.chat.completions.create(
            response_format={"type":"json_object"},
            model=self.model_name,
            messages=messages
        )
        raw_response = response.choices[0].message.content
        try:
            response = json.loads(raw_response)
        except Exception as e:
            print(f'DeepSeekClient Error: {e}')
            response = raw_response
        return response
    
class OpenAIClient(Client):
    def __init__(self, model_name, model_token) -> None:
        super().__init__(model_name, model_token)
        self.client = OpenAI(api_key=self.model_token)
    
    def inference(self, messages):
        response = self.client.chat.completions.create(
            response_format={"type":"json_object"},
            model=self.model_name,
            messages=messages
        )
        raw_response = response.choices[0].message.content
        try:
            response = json.loads(raw_response)
        except Exception as e:
            print(f'OpenAIClient Error: {e}')
            response = raw_response
        return response