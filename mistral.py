import requests

def local_ds_assistant(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "mistral",
        "prompt": f"As a Senior Data Scientist, answer this: {prompt}",
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()['response']

# Example usage
print(local_ds_assistant("Explain why I should use a YAML config for my ML pipeline."))