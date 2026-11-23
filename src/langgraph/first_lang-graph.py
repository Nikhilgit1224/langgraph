from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display
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

class NodeData(TypedDict):
    name: str
    description: str    

def welcome(sample: NodeData) -> NodeData:
    name = sample["name"]
    description = sample["description"]
    response = llm.invoke(
        f"Write one concise sentence describing {description} based on this context. "
        f"Include a greeting or repeat the name: {name}"
    ).content
    sample["description"] = f"Welcome to {name}! {response}"
    return sample    

graph = StateGraph(NodeData)
graph.add_node("welcome", welcome)
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)

first_graph = graph.compile()
state_graph = first_graph.invoke({"name": "nikhil", "description": "learning about langgraph and langchain"})
print(state_graph)
