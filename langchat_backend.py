from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests
import sqlite3
import os

load_dotenv()

model = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    openai_api_base = os.getenv("OPENAI_API_BASE")
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#--------------------------------------------Tool Creation----------------------------------------------------------#

search_tool = DuckDuckGoSearchRun(reqion="us-en")

@tool
def calc(num1: float, num2: float, op: str):
    """ Used for doing the arthematic operation on two numbers
        parameters:
            num1 : first number
            num2 : second number
            op : operations performed
    """
    try:
        if(op == "add"):
            result = num1+num2
        elif(op == "subract"):
            result = num1-num2
        elif(op == "divide"):
            if(num2 == 0):
                return{"error": "Division by zero not defined"}
            result = num1/num2
        elif(op == "multiply"):
            result = num1*num2
        else:
            return {"error": "Invalid choice"}
        
        return {"num1": num1, "num2": num2, "operation": op, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol: str)->str:
    """This Function is built for finding the stock price of a given company
    """
    api_key = os.environ["ALPHA_VANTAGE_API"]
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE&"
        f"symbol={symbol}&"
        f"apikey={api_key}"
    )

    r = requests.get(url)
    data = r.json()

    quote = data.get("Global Quote", {})
    if not quote:
        return "The stock price for this symbol is not available"
    
    return f"{quote["05. price"]}"

tools = [get_stock_price, calc, search_tool]
llm_tools = model.bind_tools(tools)

#------------------------------------------------------Graph Creation------------------------------------#

def chatnode(state: ChatState):
    query = state["messages"]
    response = llm_tools.invoke(query)
    
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chatnode", chatnode)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatnode")
graph.add_conditional_edges("chatnode", tools_condition)
graph.add_edge("tools", "chatnode")

conn = sqlite3.connect("Chatbot.db", check_same_thread = False)
checkpoint = SqliteSaver(conn = conn)

chatbot = graph.compile(checkpointer= checkpoint)

def get_db_threads():
    s = set()
    for check in checkpoint.list(None):
        thread = check.config["configurable"]["thread_id"]
        s.add(thread)
    return list(s)