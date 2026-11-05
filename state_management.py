from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the structure of our graph state
class ProductState(TypedDict):
    product_description: str
    analysis: str
    improved_description: str
    quality_score: int


# 2. Node 1: Analyze the description
def analyze(state: ProductState):
    print("\nAnalyzing product description...")

    description = state["product_description"]

    analysis = (
        "The description is too short and lacks specific "
        "product features and benefits."
    )

    print("Original description:", description)
    print("Analysis:", analysis)

    return {
        "analysis": analysis
    }


# 3. Node 2: Improve the description
def improve(state: ProductState):
    print("\nImproving product description...")

    analysis = state["analysis"]

    improved_description = (
        "Experience powerful performance with this laptop. "
        "Its fast processing capabilities make everyday "
        "tasks smoother and more efficient."
    )

    print("Based on analysis:", analysis)
    print("Improved description:", improved_description)

    return {
        "improved_description": improved_description
    }


# 4. Node 3: Review the description
def review(state: ProductState):
    print("\nReviewing improved description...")

    score = 8

    print("Description:", state["improved_description"])
    print("Quality score:", score)

    return {
        "quality_score": score
    }


# 5. Create graph builder
builder = StateGraph(ProductState)


# 6. Register nodes
builder.add_node("analyze", analyze)
builder.add_node("improve", improve)
builder.add_node("review", review)


# 7. Connect nodes
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "improve")
builder.add_edge("improve", "review")
builder.add_edge("review", END)


# 8. Compile graph
graph = builder.compile()


# 9. Provide initial state and execute
initial_state = {
    "product_description": "This laptop is good and very fast.",
    "analysis": "",
    "improved_description": "",
    "quality_score": 0
}

result = graph.invoke(initial_state)


# 10. Display final state
print("\n========== FINAL STATE ==========")

for key, value in result.items():
    print(f"{key}: {value}")