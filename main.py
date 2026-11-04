from langchain_ollama import ChatOllama

# Initialize the model

model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# Send a message to the model
response = model.invoke(
    "Explain what an LLM is in simple words."
)

# Print the generated response
print(response.content)