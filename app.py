import streamlit as st
from chains import first_responder, final_responder, global_responder, salary_responder, vacancy_responder
from langgraph.graph import MessageGraph
from classes import FinalResponse
from services.Intranet_repository import IntranetRepository
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


repository = IntranetRepository()
repository.create_or_load_faiss_index()

def create_graph():
    builder = MessageGraph()
    builder.add_node("classifier", first_responder)
    builder.add_node("global", global_responder)
    builder.add_node("salary", salary_responder)
    builder.add_node("vacancy", vacancy_responder)
    builder.add_node("final", final_responder)
    builder.add_conditional_edges("classifier", decision_flow)
    builder.add_edge("global", "final")
    builder.add_edge("salary", "final")
    builder.add_edge("vacancy", "final")
    builder.set_entry_point("classifier")
    return builder.compile()

def decision_flow(state: list[BaseMessage]) -> str:
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


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Delta Logistic Intranet Assistant")
st.write("Ask a question about the company, consult your salary, or check for vacancies by informing your code.")


graph = create_graph()
print(graph.get_graph().draw_mermaid())
if "history" not in st.session_state:
    st.session_state.history = []


query = st.chat_input("Say something")
if query:
    st.session_state.history.append(HumanMessage(content=query))
    MAX_HISTORY = 10
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    try:
        response = graph.invoke(st.session_state.history)

        final_result_json = response[-1].content
        final_result_pydantic = FinalResponse.model_validate_json(final_result_json)
        answer = final_result_pydantic.answer

        st.session_state.history.append(AIMessage(content=answer))
    except Exception as e:
        st.session_state.history.append(AIMessage(content=f"An error occurred: {str(e)}"))

# Renderizar histórico no Streamlit
for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)