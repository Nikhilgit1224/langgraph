from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os


load_dotenv()

llm = init_chat_model(
    model="gemini-3.6-flash",
    model_provider="google_genai",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant that writes a 100-word blog post about the importance of data privacy and security in the digital age."),
     ("user", "Write a 100-word blog post about {topic}.")]
)   

chain = prompt_template | llm

print(chain.invoke({"topic": "importance of ML in healthcare"}).content)