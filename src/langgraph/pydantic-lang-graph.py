from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os


load_dotenv()

llm = init_chat_model(
    model="gemini-3.6-flash",
    model_provider="google_genai",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

class LangGraphConfig(BaseModel):

    topic : str = Field(description="importance of data privacy and security in the digital age")
    post : str = Field(description="A blog post")
    curated_post : str = Field(description="A curated blog post about the importance of data privacy and security")

def to_post(state: LangGraphConfig) -> dict[str, str]:
    topic = state.topic
    response = llm.invoke(
        f"Write a 100-word blog post about {topic}."
    ).content
    return {"post": f"this is a 100-word blog post about {topic}: {response}"}

def to_curated_post(state: LangGraphConfig) -> dict[str, str]:
    post = state.post
    response = llm.invoke(
        f"Curate the following blog post to make it more engaging and informative: {post}"
    ).content
    return {"curated_post": response}

graph = StateGraph(LangGraphConfig)
graph.add_node("to_post", to_post)
graph.add_node("to_curated_post", to_curated_post)
graph.add_edge(START, "to_post")
graph.add_edge("to_post", "to_curated_post")
graph.add_edge("to_curated_post", END)

pydantic_graph = graph.compile()
print(pydantic_graph.invoke({"topic": "importance of ML in healthcare", "post": "", "curated_post": ""}))
