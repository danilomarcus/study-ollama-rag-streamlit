import ollama

models = ollama.list()

response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])

print(response['message']['content'], end='\n', flush=True)
