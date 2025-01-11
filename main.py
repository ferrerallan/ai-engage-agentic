import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessageGraph, END
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables
load_dotenv()

# Define prompts
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable expert who verifies the truthfulness of statements. Provide detailed feedback on the accuracy of the user's statement and correct any mistakes.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers any questions to the best of your ability. Provide clear and concise answers.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Configure LLM model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
general_chain = general_prompt | llm
reflect_chain = reflection_prompt | llm

# Define nodes for the graph
def general_node(messages: List[BaseMessage]):
    """Handles general questions."""
    return general_chain.invoke({"messages": messages})

def reflection_node(messages: List[BaseMessage]):
    """Handles reflection/verification of user input."""
    statement = messages[-1].content
    res = reflect_chain.invoke({"messages": [HumanMessage(content=statement)]})
    return [AIMessage(content=res.content)]

# Decision function to verify if the input is a learning instruction
def should_verify(state: List[BaseMessage]):
    """Check if the latest message is a learning instruction."""
    last_message = state[-1].content.lower()
    if last_message.startswith("learn: "):
        return "reflect"
    return "general"

# Initialize Streamlit
st.title("LangChain Chat - Interactive QA")

# Maintain chat history
if "history" not in st.session_state:
    st.session_state.history = []  # List to store message history

# User input
user_input = st.text_input("Enter your message:", key="user_input", placeholder="Ask a question or provide a learning instruction starting with 'learn:'")

if user_input:
    # Add user's message to history
    st.session_state.history.append(HumanMessage(content=user_input))

    # Build the graph
    builder = MessageGraph()
    builder.add_node("general", general_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("general")

    # Add conditional edge to decide whether to verify or continue answering general questions
    builder.add_conditional_edges("general", should_verify)
    builder.add_edge("reflect", "general")  # Return to general after reflection

    # Compile and run the graph
    graph = builder.compile()
    response = graph.invoke(st.session_state.history)

    # Add AI response to history
    st.session_state.history.append(AIMessage(content=response[-1].content))

    # Display chat response
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Assistant:** {response[-1].content}")

# Display chat history in the sidebar
st.sidebar.title("Chat History")
for msg in st.session_state.history:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    st.sidebar.markdown(f"**{role}:** {msg.content}")
