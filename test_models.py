import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

models_to_test = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-lite-latest",
]

for model in models_to_test:

    print(f"\nTesting: {model}")

    try:

        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: RAG TEST OK"
        )

        print("SUCCESS:", response.text)

    except Exception as e:

        print("FAILED:", type(e).__name__)
        print(str(e))