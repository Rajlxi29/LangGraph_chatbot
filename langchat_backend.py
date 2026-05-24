from __future__ import annotations

from typing import TypedDict, Annotated, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain_openai import ChatOpenAI
import tempfile
from dotenv import load_dotenv
import requests
import sqlite3
import os

load_dotenv()

hf_token = os.environ["HF_token"]
login(hf_token)

embedding = HuggingFaceEmbeddings(model = "sentnce-transformes/all-MiniLM-L6-v2")

_THREAD_RETRIVER: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIVER:
        return _THREAD_RETRIVER[thread_id]
    return None

def get_doc(filebytes: bytes, thread_id: str, filename: Optional[str]=None):

    if not filebytes:
        raise ValueError("No bytes recieved from ingestion")
    
    with tempfile.NamedTemproryFile(delete=False, suffix=".pdf") as tempfile:
        tempfile.write(filebytes)
        file_path = tempfile.name

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=100, separators=["\n","\n\n"," "])
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(documents = chunks, embedding = embedding)
        retriver = vectorstore.as_retriever(search_type='similarity', kwargs={'k':4})

        _THREAD_RETRIVER[str(thread_id)] = retriver
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(file_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        return {
            "filename": filename or os.path.basename(file_path),
            "document": len(docs),
            "chunks": len(chunks)
        }
    
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass




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

@tool
def rag(query: str, thread_id: Optional[str] = None):
    
    retrieve = get_retriever(thread_id)

    if retrieve is None:
        return {
            "error": "Upload a document to get data",
            "query": query
        }
    
    result = retrieve.invoke(query)
    context = [docs.page_content for docs in result]
    metadata = [docs.metadata for docs in result]

    return{
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }

tools = [get_stock_price, calc, search_tool]
llm_tools = model.bind_tools(tools)



#------------------------------------------------------Graph Creation------------------------------------#

def chatnode(state: ChatState, config = None):

    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    
    system = SystemMessage(content= 
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    
    messages = [system, state["messages"]]

    response = llm_tools.invoke(messages, cofig = config)
    
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