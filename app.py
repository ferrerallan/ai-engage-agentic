import streamlit as st
from chains import first_responder, final_responder, global_responder, salary_responder, vacancy_responder
from langgraph.graph import MessageGraph
from classes import FinalResponse
from services.Intranet_repository import IntranetRepository
import json
from langchain_core.messages import BaseMessage, HumanMessage


repository = IntranetRepository()
repository.create_or_load_faiss_index()

def create_graph():
    builder = MessageGraph()
    builder.add_node("draft", first_responder)
    builder.add_node("global", global_responder)
    builder.add_node("salary", salary_responder)
    builder.add_node("vacancy", vacancy_responder)
    builder.add_node("final", final_responder)
    builder.add_conditional_edges("draft", event_loop)
    builder.add_edge("global", "final")
    builder.add_edge("salary", "final")
    builder.add_edge("vacancy", "final")
    builder.set_entry_point("draft")
    return builder.compile()

def event_loop(state: list[BaseMessage]) -> str:
    last_message = state[-1]
    if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs['tool_calls']
        if tool_calls:
            last_tool = tool_calls[-1]
            arguments = last_tool['function']['arguments']
            result = json.loads(arguments)
            if result.get("request_type") == "global_question":
                return "global"
            elif result.get("request_type") == "salary_request":
                return "salary"
            elif result.get("request_type") == "vacancy_request":
                return "vacancy"
    return "final"

def format_message_history(history):
    return "\n".join([f"User: {msg.content}" for msg in history])


# Streamlit interface
st.title("Delta Logistic Intranet Assistant")
st.write("Ask a question about company or consult your salary or vacancies, by informing your code.")



query = st.text_input("Enter your question:")
if query:
    graph = create_graph()
    try:
        response = graph.invoke(HumanMessage(content=query))
        final_result_json = response[-1].content
        final_result_pydantic = FinalResponse.model_validate_json(final_result_json)
        st.success(f"Response: {final_result_pydantic.answer}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
