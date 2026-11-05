
from typing import Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama


# 1. Define the expected output structure
class ProductReview(BaseModel):
    quality: Literal["good", "bad"]

    score: int = Field(
        ge=1,
        le=10,
        description="Product description quality score"
    )

    reason: str


# 2. Initialize the Ollama model
model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)


# 3. Configure structured output
structured_model = model.with_structured_output(
    ProductReview
)


# 4. Ask the model to analyze a description
result = structured_model.invoke(
    """
    Analyze this product description:

    This laptop has 16GB RAM,
    a powerful processor, and a lightweight design.
    """
)


# 5. Access the structured result
print(result)

print("Quality:", result.quality)
print("Score:", result.score)
print("Reason:", result.reason)
