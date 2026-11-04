from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. Configure the local model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# 2. Create a reusable prompt template
prompt_template = PromptTemplate.from_template(
    "Explain {topic} in simple words for a {audience}."
)

# 3. Insert values into the template
prompt = prompt_template.format(
    topic="LangChain",
    audience="beginner"
)

# 4. Send the completed prompt to the model
response = model.invoke(prompt)

# 5. Display the answer
print(response.content)