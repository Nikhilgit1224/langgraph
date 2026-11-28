import os
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END



load_dotenv()

llm = init_chat_model(
    model="gemini-3.6-flash",
    model_provider="google_genai",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# ************** DuckDuckGo Search Tool ****************
@tool
def duckduckgo_tool(query: str) -> str:
    '''this is the function that takes query as input and returns a string with the search results'''
    duckduckgo_tool = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper(
            region="us-en",
            backend="google,bing,brave,yahoo",
            time=None,
        )
    )
    return duckduckgo_tool.invoke(query)


# ************** Wikipedia Search Tool ****************
@tool
def wiki_tool(query: str) -> str:
    '''this is the function that takes query as input and returns a string with the search results'''
    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper()
    )
    return wikipedia_tool.invoke(query)


# ************** Arxiv Search Tool **************** 
@tool
def arxiv_tool(query: str) -> str:
    '''this is the function that takes query as input and returns a string with the search results'''
    arxiv_query_tool = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(),
    )
    return arxiv_query_tool.invoke(query)

# ************** Custom Tool ****************
@tool
def personal_info(name:str) -> str:

    '''this is the function that takes name as input and returns a string with the information'''
    info = {
        "alice": "Alice is a software engineer.",
        "sam": "Sam is a data scientist.",
        "bob": "Bob is a product manager."
    }
    return info.get(name.lower(), "No information available for this name.")

tools = [duckduckgo_tool, wiki_tool, arxiv_tool, personal_info]

llm_with_tools = llm.bind_tools(tools)

response_llm = llm_with_tools.invoke("what will alice do?")
#print(response_llm.tool_calls)

# *************** Lang graph creation ****************

class graph_schema(TypedDict):
    messages : list

def if_tool_call(state :graph_schema) -> graph_schema:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_call"
    else:
        return END

def llm_node(state: graph_schema) -> graph_schema:
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant that can use tools to answer questions."),
         ("user", "{input}")]
    )

    chain = prompt_template | llm_with_tools
    response = chain.invoke({"input": messages})
    state["messages"] = messages + [response]

    return state


def tool_node(state: graph_schema) -> graph_schema:
    messages = state["messages"]
    tools_by_name = {tool.name: tool for tool in tools}
    tool_results = []

    for tool_call in messages[-1].tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool is None:
            observation = f"Unknown tool: {tool_call['name']}"
        else:
            observation = tool.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    state["messages"] = tool_results
    return state


graph = StateGraph(graph_schema)
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "llm_node")
graph.add_conditional_edges("llm_node", if_tool_call, {"tool_call": "tool_node", END: END})
graph.add_edge("tool_node", "llm_node")
graph.add_edge("llm_node", END)

graph_with_tools = graph.compile()

response_graph = graph_with_tools.invoke({"messages": [HumanMessage(content="what is gold rate today")]})

print(response_graph)
