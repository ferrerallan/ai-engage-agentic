from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import BaseModel
from langchain.tools import BaseTool

# Define a Pydantic model for the input
class QuestionInput(BaseModel):
    question: str

# Define a tool that uses the input model
class AnswerQuestion(BaseTool):
    name: str = "answer_question"  # Add type annotation
    description: str = "A tool to answer a question"  # Add type annotation for description

    def _run(self, question: str) -> str:  # Ensure correct method name and signature
        return f"Answering the question: {question}"

    async def _arun(self, question: str) -> str:  # For async support if needed
        raise NotImplementedError("Async functionality is not implemented.")

# Instantiate the tool
answer_question_tool = AnswerQuestion()

# Create a parser and pass the tool
parser_pydantic = PydanticToolsParser(tools=[answer_question_tool])

# Example usage in a LangChain pipeline
response = {
    "tool": "answer_question",
    "tool_input": {"question": "What is LangChain?"},
}

# Parse and validate the response
validated_tool = parser_pydantic.parse(response)

# Execute the tool logic
result = validated_tool.run(validated_tool.tool_input["question"])

print(result)



first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)