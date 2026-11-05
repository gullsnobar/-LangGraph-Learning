from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# 1. Define a Python function as a tool
@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b


# 2. Initialize the local LLM
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)


# 3. Give the model access to our tool
model_with_tools = model.bind_tools([multiply_numbers])


# 4. Ask a question
question = "Use the multiply_numbers tool to calculate 25 multiplied by 3."

# 5. Send the question to the model
ai_response = model_with_tools.invoke(question)

print("Tool calls requested by the model:")
print(ai_response.tool_calls)


# 6. Execute the requested tool calls
if not ai_response.tool_calls:
    print("The model did not request a tool call.")
else:
    messages = [
        HumanMessage(content=question),
        ai_response
    ]

    available_tools = {
        "multiply_numbers": multiply_numbers
    }

    for tool_call in ai_response.tool_calls:
        selected_tool = available_tools[tool_call["name"]]

        result = selected_tool.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    # 7. Send the tool result back to the model
    final_response = model_with_tools.invoke(messages)

    print("\nFinal answer:")
    print(final_response.content)