from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the chat model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# Define conversation messages
messages = [
    SystemMessage(
        content="You are a Python teacher. Explain everything simply."
    ),
    HumanMessage(
        content="What is a Python variable?"
    )
]

# Send messages to the model
response = model.invoke(messages)

# Display AI response
print(response.content)