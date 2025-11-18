import os
import requests
from dotenv import load_dotenv

load_dotenv()  # only if you're using .env

api_key = os.getenv("OPENROUTER_API_KEY")  # or paste it directly if testing
if not api_key:
    print("API key not found.")
    exit()

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://your-site.com",  # or your app domain
    "X-Title": "Model Checker",
}
response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)

if response.ok:
    models = response.json().get("data", [])
    print("✅ You can access the following models:\n")
    for model in models:
        print("-", model.get("id"))
else:
    print("❌ Error:", response.text)