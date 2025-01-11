import streamlit as st
from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph
from chains import answer_chain,  ask_llm

load_dotenv()

START="start"
ANSWER = "answer"
REFLECT = "reflect"

def answer_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = answer_chain.invoke({"messages": state})
    return [AIMessage(content=res.content)]

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    statement = user_messages[-1].content
    print(statement)
    res = ask_llm(statement)
    print("Reflection response:", res)
    
    if "false" in res.lower():
        return [AIMessage(content=f"The statement '{statement}' is not verifiable globally and is likely false.")]
    elif "true" in res.lower():
        return [AIMessage(content=f"Thanks, I learned it!")]
    else:
        return [AIMessage(content=f"The statement '{statement}' is not verifiable globally.")]

def should_verify(state: List[BaseMessage]):
    user_messages = [msg for msg in state if isinstance(msg, HumanMessage)]
    
    if not user_messages:
        return END

    last_message = user_messages[-1].content.lower()
    print("Start Node - Last user message:", last_message)

    if last_message.startswith("learn: "):
        print("Redirecting to REFLECT.")
        return REFLECT

    print("Redirecting to ANSWER.")
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

# Render input box
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
