import os
from openai import OpenAI

def init_openai(api_path='../openai_api.txt'):
    with open(api_path, 'r') as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

init_openai()
client = OpenAI()

file_response = client.files.content("file-gEGDZdwSfYLk4ptic42FTSiV")
print(file_response.text)