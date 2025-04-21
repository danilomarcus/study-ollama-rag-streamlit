import requests
import json
from pprint import pprint

url = "http://127.0.0.1:11434/api/generate"

data = {
    "model": "llama3.2",
    "prompt": "Tell me a joke, responda apenas em português brasileiro"
}

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    try:
        # Parse the JSON response
        for line in response.iter_lines():
            if line:
                text = ""
                if not json.loads(line)['done']:
                    text += json.loads(line)['response']
                
                print(text, end='', flush=True)

        print("\n")                
    except json.JSONDecodeError:
        print("A resposta não está em formato JSON válido:")
        print(response.text)
else:
    print(f"Erro: {response.status_code}")
    print(response.text)
