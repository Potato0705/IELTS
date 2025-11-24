import os, requests, json, textwrap

API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
assert API_KEY, "No OPENROUTER_API_KEY env!"

BASE = "https://openrouter.ai"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://ielts-alphaevolve.local",
    "X-Title": "IELTS-Scoring",
}

def show(name, r):
    print(f"\n=== {name} ===")
    print("status:", r.status_code)
    print("url:", r.url)
    print("text:", r.text[:400])
    print("repr(key) head/tail:", repr(API_KEY[:12]), repr(API_KEY[-12:]))
    print("has whitespace ends?", API_KEY[:1].isspace(), API_KEY[-1:].isspace())
# 1) 最基础的 GET：列模型（只要这个都 404/被劫持，就说明不是代码问题）
r = requests.get(BASE + "/api/v1/models", headers=headers, timeout=30)
show("GET /models", r)

# 2) Chat Completions
payload_chat = {
    "model": "deepseek/deepseek-r1:free",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 8,
}
r = requests.post(BASE + "/api/v1/chat/completions", headers=headers, json=payload_chat, timeout=30)
show("POST /chat/completions", r)

# 3) Responses Beta
payload_resp = {
    "model": "deepseek/deepseek-r1:free",
    "input": "Hello",
    "max_output_tokens": 8,
}
r = requests.post(BASE + "/api/v1/responses", headers=headers, json=payload_resp, timeout=30)
show("POST /responses", r)

