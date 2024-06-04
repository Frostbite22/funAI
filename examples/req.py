import requests

url = "http://127.0.0.1:8000/generate"
payload = {
    "prompt": "Once upon a time",
    "max_tokens": 50
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
