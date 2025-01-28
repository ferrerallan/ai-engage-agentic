import os
from dotenv import load_dotenv
import datetime
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests
from classes import ClassifyQuestion, FinalResponse, GlobalResponse, SalaryResponse, VacancyResponse
from langchain_core.messages import ToolMessage
import json
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from services.Intranet_repository import IntranetRepository

load_dotenv()
llm = ChatOpenAI(model="gpt-4-turbo-preview")

intranet_repository = IntranetRepository()
vectorstore = intranet_repository.create_or_load_faiss_index()

### classifier ###
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI specialized in analyzing and extracting structured information from text.

            Current time: {time}

            Instructions:
            1. Analyze the user's input and classify it into one of the following categories:
            - 'salary_request': Queries related to salary.
            - 'vacancy_request': Queries related to leave balances.
            - 'global_question': General or unrelated queries.
            2. Identify and extract the 'employeeCode' if present. The employee code can appear in various forms, including but not limited to:
            - Phrases like "my number is xxx ", "my code is xxx", "employee ID is xxx", "ID: xxx", or any similar variation.
            - Formats such as numeric codes (e.g., 12345) or alphanumeric (e.g., ABC123).
            3. Return the response in JSON format with the following structure:
            - 'request_type': The identified type of request.
            - 'employee_code': The extracted employee code, or null if none is found.
            
            Be flexible in recognizing variations of phrases and contexts, ensuring high accuracy in classification and code extraction..""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the last user's question above using the required format, and considering history when needed."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder = actor_prompt_template | llm.bind_tools(
    tools=[ClassifyQuestion], tool_choice="ClassifyQuestion"
)

### Final ###
def final_responder(input_messages):
    last_message = input_messages[-1]
    if hasattr(last_message, 'additional_kwargs') and \
                'tool_calls' in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs['tool_calls']
        last_tool = tool_calls[-1]
        arguments = last_tool['function']['arguments']
        result = json.loads(arguments)
        tool_name = last_tool['function']['name']
        if tool_name == 'ClassifyQuestion':
            answer = result['request_type'].upper()
            if 'employee_code' in result:
                answer += f" ({result['employee_code']})"
        elif tool_name in ['GlobalResponse', 'SalaryResponse', 'VacancyResponse']:
            answer = result['answer']
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        final_response = FinalResponse(answer=answer)
        return ToolMessage(
            name="FinalResponder",
            content=final_response.json(),
            tool_call_id=last_tool['id']
        )
    raise ValueError("No valid message found.")


### Global ###
def query_document(question, vectorstore, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    if docs:
        return "\n".join([doc.page_content for doc in docs])
    return "No relevant information found."

def build_prompt_with_context(question, context):
    prompt = f"""
    You are an expert assistant. Use the context below to answer the
      user's question accurately and concisely.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt

def global_responder_logic(input_message):
    last_human_message = None
    for message in reversed(input_message):
        if isinstance(message, HumanMessage): 
            last_human_message = message.content
            break

    if not last_human_message:
        raise ValueError("No human message found in the input messages.")

    # Construir contexto e criar resposta
    context = query_document(last_human_message, vectorstore)
    prompt = build_prompt_with_context(last_human_message, context)
    response = llm.predict(prompt)
    global_response = GlobalResponse(answer=response)
    return global_response.json()

global_responder = global_responder_logic | llm.bind_tools(
    tools=[GlobalResponse], tool_choice="GlobalResponse"
)

### salary ###
def salary_responder_logic(input_message):
    if hasattr(input_message[-1], 'additional_kwargs') and \
        'tool_calls' in input_message[-1].additional_kwargs:
        tool_calls = input_message[-1].additional_kwargs['tool_calls']
        last_tool = tool_calls[-1]
        arguments = last_tool['function']['arguments']
        result = json.loads(arguments)
        employee_code = result.get("employee_code")
        if not employee_code:
            raise ValueError("employee_code not found.")
    else:
        raise ValueError("No valid message found to extract employee_code.")
    url = os.getenv("SALARY_ENDPOINT_URL")
    payload = {"employeeCode": employee_code}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        api_result = response.json()
        
        salary_days = api_result.get('YTDPayroll', '-1')

        if salary_days == -1:
            message = "No data available for this employee."
        else:
            name = api_result.get('name')
            message = f"{name}, your YTD salary is {salary_days}"
        salary_response = SalaryResponse(answer=message)
    except requests.RequestException as e:
        salary_response = SalaryResponse(answer=f"Error: {str(e)}")
    return salary_response.json()

salary_responder = salary_responder_logic | llm.bind_tools(
    tools=[SalaryResponse], tool_choice="SalaryResponse"
)


### vacancy ###
def vacancy_responder_logic(input_message):
    if hasattr(input_message[-1], 'additional_kwargs') and \
        'tool_calls' in input_message[-1].additional_kwargs:
        tool_calls = input_message[-1].additional_kwargs['tool_calls']
        last_tool = tool_calls[-1]
        arguments = last_tool['function']['arguments']
        result = json.loads(arguments)
        employee_code = result.get("employee_code")
        if not employee_code:
            raise ValueError("employee_code not found.")
    else:
        raise ValueError("No valid message found to extract employee_code.")
    url = os.getenv("VACANCY_ENDPOINT_URL")
    
    payload = {"employeeCode": employee_code}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        api_result = response.json()

        vacancy_days = api_result.get('vacancyBalanceDays', '-1')

        if vacancy_days == -1:
            message = "No data available for this employee."
        else:
            name = api_result.get('name')
            message = f"Your vacancy balance days is {vacancy_days}, {name}. Enjoy your time off!"
        vacancy_response = VacancyResponse(answer=message)
    except requests.RequestException as e:
        vacancy_response = VacancyResponse(answer=f"Error: {str(e)}")
    return vacancy_response.json()

vacancy_responder = vacancy_responder_logic | llm.bind_tools(
    tools=[VacancyResponse], tool_choice="VacancyResponse"
)
