from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 1
builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()

print(graph.get_graph().draw_mermaid())


def process_final_result(res):
    
    final_message = res[-1]  
    final_answer = final_message.tool_calls[0]["args"]["answer"]  

    references = final_message.tool_calls[0]["args"].get("references", [])
    formatted_references = "\n".join(references) if references else "No references provided."

    final_output = f"{final_answer}\n\nReferences:\n{formatted_references}"
    return final_output



res = graph.invoke(
    "make a comparison between full stack engineers and AI Engineers,"
    " focusing in a prediction about demands for the next 10 years"
)


print("\n\n")
print("***********************************************")
print("\n\n")
print(process_final_result(res))
