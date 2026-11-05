from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# Create prompt
prompt = PromptTemplate.from_template("""
Analyze this product description:

{description}

Return ONLY valid JSON with these three fields:
- quality: good or bad
- score: a number from 1 to 10
- reason: a short explanation

Do not include markdown or additional text.
""")

# Create parser
parser = JsonOutputParser()

# Connect everything
chain = prompt | model | parser

# Execute
result = chain.invoke({
    "description": "This laptop has 16GB RAM, a fast processor, and a lightweight design."
})

print(result)
print("Quality:", result["quality"])
print("Score:", result["score"])