from langgraph.graph import MessageGraph
from chains import first_responder, final_responder, global_responder, salary_responder, vacancy_responder
from classes import FinalResponse
from langchain_core.messages import BaseMessage
import json
from dotenv import load_dotenv
from services.Intranet_repository import IntranetRepository

load_dotenv()

repository = IntranetRepository()
repository.create_or_load_faiss_index()

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

builder = MessageGraph()
builder.add_node("classifier", first_responder)
builder.add_node("global", global_responder)
builder.add_node("salary", salary_responder)
builder.add_node("vacancy", vacancy_responder)
builder.add_node("final", final_responder)
builder.add_conditional_edges("classifier", event_loop)
builder.add_edge("global", "final")
builder.add_edge("salary", "final")
builder.add_edge("vacancy", "final")
builder.set_entry_point("draft")
graph = builder.compile()

try:
    response = graph.invoke("tell me about my earnings. my code is abc123")
    final_result_json = response[-1].content
    final_result_pydantic = FinalResponse.model_validate_json(final_result_json)
    print("Final Result (Pydantic):", final_result_pydantic.answer)
except Exception as e:
    print("Error during graph execution:", e)
