
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

llm = init_chat_model(
    model="gemini-3.5-flash",
    model_provider="google_genai",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

response = llm.invoke("Explain Retrieval Augmented Generation (RAG).")

print(response.content)