
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Configure the local model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# Step 2: Create a reusable chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming teacher."),
    ("human", "Explain {topic} in simple words.")
])

# Step 3: Connect the components into a chain
chain = prompt | model

# Step 4: Run the chain with input data
response = chain.invoke({
    "topic": "Python functions"
})

# Step 5: Display the result
print(response.content)
