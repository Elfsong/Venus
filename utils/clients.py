# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/08/31

import os
from openai import OpenAI

class Client(object):
    def __init__(self, model_name, model_token) -> None:
        self.model_name = model_name
        self.model_token = model_token
        self.client = None
    
    def inference(self, context):
        raise NotImplementedError("Don't call the base class directly.")
    
    
class OpenAIClient(Client):
    def __init__(self, model_name, model_token) -> None:
        super().__init__(model_name, model_token)
        self.client = OpenAI(api_key=self.model_token)
    
    def inference(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    model_token = os.getenv("OPENAI_TOKEN")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a LLM?"}
    ]
    client = OpenAIClient(model_name="gpt-4o", model_token=model_token)
    response = client.inference(messages)
    print(response)
        
        