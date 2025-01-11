import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define prompts
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            " Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Configure LLM model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

# Initialize Streamlit
st.title("LangChain Chat - Interactive Conversation")

# Maintain chat history
if "history" not in st.session_state:
    st.session_state.history = []  # List to store message history

# User input
user_input = st.text_input("Enter your message:", key="user_input", placeholder="Type something here...")

# Chat logic
if user_input:
    # Add user message to history
    st.session_state.history.append(HumanMessage(content=user_input))

    # Check if the user wants to end the conversation
    if user_input.strip().lower() == "bye":
        st.markdown("**Chat ended. Thank you for participating!**")
    else:
        # Process message using the generate_chain
        ai_response = generate_chain.invoke({"messages": st.session_state.history})
        st.session_state.history.append(AIMessage(content=ai_response.content))

        # Display chat response
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Assistant:** {ai_response.content}")

# Display chat history in the sidebar
st.sidebar.title("Chat History")
for msg in st.session_state.history:
    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
    st.sidebar.markdown(f"**{role}:** {msg.content}")
