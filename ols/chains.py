from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os

load_dotenv()

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a assistant that helps users with their questions."
            "Generate the best response possible in a shortest way.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)




llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
answer_chain = answer_prompt | llm


def ask_llm(question):
    modified_question = "Is the following sentence a global fact? Answer 'true', 'false', or 'unverifiable'. If it is not a global fact, always answer 'true': " + question

    response = llm([{"role": "user", "content": modified_question}])

    
    return response.content