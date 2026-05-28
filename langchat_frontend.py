import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchat_backend import chatbot, get_db_threads, get_doc, thread_document_metadata
import uuid

#session states for message history and thread_id 

def generate_thread():
    thread_id = uuid.uuid4()
    return thread_id

def new_chat():
    thread_id = generate_thread()
    st.session_state["thread_id"] = thread_id
    add_chat(thread_id)
    st.session_state["message_history"] = []

def add_chat(thread_id):
    if thread_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"].append(thread_id)

def get_thread(thread_id):
    state = chatbot.get_state(config = {"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = get_db_threads()

if "ingest_document" not in st.session_state:
    st.session_state["ingest_document"] = {}

add_chat(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingest_document"].setdefault(thread_key,{})
threads = st.session_state["chat_history"][::-1]
selected_thread = None


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["messages"])

user_input = st.chat_input("Ask anything")

st.sidebar.title("Langchat")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using the document {latest_doc["filename"]} in the chat \n"
        f"{latest_doc.get('chunks')} chunks from {latest_doc.get('pages')} pages"
    )
else:
    st.sidebar.info("No pdf Indexed yet")

uploaded_file = st.sidebar.file_uploader("Upload your file here", type=["pdf"])



if uploaded_file:
    if uploaded_file.name in thread_docs:
        st.sidebar.info(f"{uploaded_file.name} is being processed")
    else:
        with st.sidebar.status("Indexing...", expanded= True) as status_box:
            summary = get_doc(
                uploaded_file.getvalue(),
                thread_id = thread_key,
                filename = uploaded_file.name,
            )
            thread_docs[uploaded_file.name] = summary
            status_box.update(label="File Indexed", state="complete", expanded = False)


if st.sidebar.button("New chat"):
    new_chat()
    st.rerun()

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id



if selected_thread:
    temp_mess = []
    role = ""

    for mess in get_thread(selected_thread):
        if isinstance(mess, HumanMessage):
                role = "user"
        else:
            role = "assistant"
        temp_mess.append({"role": role, "messages": mess.content})

        
    st.session_state["message_history"] = temp_mess
    st.rerun()

#For different threads
configure = {
    "configurable":{"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn"
}

if user_input:

    st.session_state["message_history"].append({"role": "user", "messages": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
            
        def ai_message():
            for message_chunk ,metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config = configure,
                stream_mode = "messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_mess = st.write_stream(ai_message())
        
    st.session_state["message_history"].append({"role": "assistant", "messages": ai_mess})
    