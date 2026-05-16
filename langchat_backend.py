from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

model = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    openai_api_base = os.getenv("OPENAI_API_BASE")
)

class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]

def chatnode(state: ChatState):
    query = state["message"]
    response = model.invoke(query)
    
    return {"message": [response]}

graph = StateGraph(ChatState)

graph.add_node("chatnode", chatnode)

graph.add_edge(START, "chatnode")
graph.add_edge("chatnode", END)

conn = sqlite3.connect("Chatbot.db", check_same_thread = False)
checkpoint = SqliteSaver(conn = conn)

chatbot = graph.compile(checkpointer= checkpoint)

def get_db_threads():
    s = set()
    for check in checkpoint.list(None):
        thread = check.config["configurable"]["thread_id"]
        s.add(thread)
    return list(s)