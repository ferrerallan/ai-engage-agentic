import streamlit as st
from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from chromadb import HttpClient
import os

from chains import ask_llm, answer_chain

load_dotenv()

START = "start"
ANSWER = "answer"
REFLECT = "reflect"

# ChromaDB configuration
api_key = os.getenv("OPENAI_API_KEY")
chroma_client = HttpClient(host="localhost", port=8001)
vectorstore = Chroma(
    client=chroma_client,
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key)
)

def save_to_db(statement: str):
    try:
        vectorstore.add_texts([statement])
        print("Knowledge stored successfully!")
    except Exception as e:
        print(f"Error storing knowledge: {e}")

def query_db(question: str):
    try:
        results = vectorstore.similarity_search(question, k=3)
        return [doc.page_content for doc in results] if results else []
    except Exception as e:
        print(f"Error retrieving knowledge: {e}")
        return []

def answer_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    question = state[-1].content if state else ""
    learned_info = query_db(question)
    combined_question = question + "\nLearned information: " + "\n".join(learned_info)
    res = answer_chain.invoke({"messages": [HumanMessage(content=combined_question)]})
    return [AIMessage(content=res.content)]

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    statement = user_messages[-1].content
    res = ask_llm(statement)

    if "false" in res.lower():
        return [AIMessage(content=f"The statement '{statement}' is not verifiable globally and is likely false.")]
    elif "true" in res.lower():
        save_to_db(statement)
        return [AIMessage(content=f"Thanks, I learned it!")]
    else:
        return [AIMessage(content=f"The statement '{statement}' is not verifiable globally.")]

def should_verify(state: List[BaseMessage]):
    user_messages = [msg for msg in state if isinstance(msg, HumanMessage)]

    if not user_messages:
        return END

    last_message = user_messages[-1].content.lower()

    if last_message.startswith("learn: "):
        return REFLECT

    return ANSWER

def start_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    return []

builder = MessageGraph()
builder.add_node(START, start_node)
builder.add_node(ANSWER, answer_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(START)

builder.add_conditional_edges(START, should_verify)
builder.add_edge(REFLECT, END)
builder.add_edge(ANSWER, END)

graph = builder.compile()

st.title("LangChain Chat - Interactive QA")
st.sidebar.title("Chat History")

if "history" not in st.session_state:
    st.session_state.history = []

if "input" not in st.session_state:
    st.session_state.input = ""

user_input = st.text_input(
    "Enter your message:",
    value=st.session_state.input,
    placeholder="Ask a question or provide a learning instruction starting with 'learn:'",
    key="user_input",
)

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.input = ""

    last_message = [HumanMessage(content=user_input)]
    responses = graph.invoke(last_message)

    for response in responses:
        if isinstance(response, AIMessage):
            st.session_state.history.append(response)
        elif response == END:
            break

for msg in st.session_state.history:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    st.markdown(f"**{role}:** {msg.content}")

for msg in st.session_state.history:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    st.sidebar.markdown(f"**{role}:** {msg.content}")
