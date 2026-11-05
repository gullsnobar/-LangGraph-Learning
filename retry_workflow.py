from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define graph state
class ProductState(TypedDict):
    product_description: str
    analysis: str
    improved_description: str
    quality_score: int
    retry_count: int
    max_retries: int
    status: str


# 2. Analyze node
def analyze(state: ProductState):
    print("\n[ANALYZE] Analyzing description...")

    return {
        "analysis": (
            "The description needs more specific "
            "features and customer benefits."
        )
    }


# 3. Improve node
def improve(state: ProductState):
    attempt = state["retry_count"] + 1

    print(f"\n[IMPROVE] Improvement attempt: {attempt}")

    improved_description = (
        "Discover reliable performance with this laptop. "
        "Designed for productivity, it delivers a smooth "
        "experience for work, study, and entertainment. "
        "Its practical design makes everyday computing easier."
    )

    return {
        "improved_description": improved_description,
        "retry_count": attempt
    }


# 4. Review node
def review(state: ProductState):
    print("\n[REVIEW] Evaluating description...")

    # Simulated scores for learning purposes.
    # Attempt 1 = 6, attempt 2 = 7, attempt 3 = 8.
    score = min(5 + state["retry_count"], 8)

    print("Current score:", score)

    return {
        "quality_score": score
    }


# 5. Routing function
def route_review(state: ProductState):

    if state["quality_score"] >= 8:
        return "approved"

    if state["retry_count"] < state["max_retries"]:
        return "retry"

    return "max_retries_reached"


# 6. Approved node
def approved(state: ProductState):
    print("\n[APPROVED] Content meets quality requirements.")

    return {
        "status": "approved"
    }


# 7. Failure node
def max_retries_reached(state: ProductState):
    print("\n[STOPPED] Maximum retry attempts reached.")

    return {
        "status": "needs_manual_review"
    }


# 8. Create graph
builder = StateGraph(ProductState)


# 9. Register nodes
builder.add_node("analyze", analyze)
builder.add_node("improve", improve)
builder.add_node("review", review)
builder.add_node("approved", approved)
builder.add_node("max_retries_reached", max_retries_reached)


# 10. Connect nodes
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "improve")
builder.add_edge("improve", "review")


# 11. Conditional routing
builder.add_conditional_edges(
    "review",
    route_review,
    {
        "approved": "approved",
        "retry": "improve",
        "max_retries_reached": "max_retries_reached"
    }
)


# 12. End both possible paths
builder.add_edge("approved", END)
builder.add_edge("max_retries_reached", END)


# 13. Compile graph
graph = builder.compile()


# 14. Initial state
initial_state = {
    "product_description": "This laptop is good and very fast.",
    "analysis": "",
    "improved_description": "",
    "quality_score": 0,
    "retry_count": 0,
    "max_retries": 3,
    "status": "pending"
}


# 15. Execute graph
result = graph.invoke(initial_state)


# 16. Display final result
print("\n========== FINAL RESULT ==========")
print("Quality score:", result["quality_score"])
print("Retry count:", result["retry_count"])
print("Status:", result["status"])