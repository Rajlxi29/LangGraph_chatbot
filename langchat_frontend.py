import streamlit as st
from langchain_core.messages import HumanMessage
from langchat_backend import chatbot, get_db_threads
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
    return state.values.get("message", [])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = get_db_threads()

add_chat(st.session_state["thread_id"])

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["message"])

user_input = st.chat_input("Ask anything")

st.sidebar.title("Langchat")

if st.sidebar.button("New chat"):
    new_chat()
    st.rerun()

st.sidebar.title("Chats")
for thread in st.session_state["chat_history"]:
    if st.sidebar.button(str(thread)):
        temp_mess = []
        role = ""

        for mess in get_thread(thread):
            if isinstance(mess, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_mess.append({"role": role, "message": mess.content})

        
        st.session_state["message_history"] = temp_mess
        st.rerun()

config = {
    "configurable":{"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id":st.session_state["thread_id"]},
    "run_name": "chat_turn"
}

if user_input:

    st.session_state["message_history"].append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        output = st.write_stream(
            message_chunk.content for message_chunk ,metadata in chatbot.stream(
                {"message": [HumanMessage(content=user_input)]},
                config = config,
                stream_mode = "messages"
            )
        )
        
    st.session_state["message_history"].append({"role": "assistant", "message": output})
    