from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define our graph state
class ProductState(TypedDict):
    product_description: str
    analysis: str
    improved_description: str
    quality_score: int
    status: str


# 2. Node: Analyze
def analyze(state: ProductState):
    print("\n[ANALYZE] Analyzing product description...")

    analysis = (
        "The description needs more details about "
        "product features and benefits."
    )

    return {
        "analysis": analysis
    }


# 3. Node: Improve
def improve(state: ProductState):
    print("\n[IMPROVE] Improving product description...")

    improved_description = (
        "Discover powerful performance with this laptop. "
        "Designed for productivity, it offers a smooth "
        "computing experience for everyday tasks, work, "
        "and entertainment."
    )

    return {
        "improved_description": improved_description
    }


# 4. Node: Review
def review(state: ProductState):
    print("\n[REVIEW] Reviewing product description...")

    description = state["improved_description"]

    # Simple rule-based score for learning purposes
    score = 8 if len(description) >= 100 else 5

    print("Quality score:", score)

    return {
        "quality_score": score
    }


# 5. Routing function
def route_review(state: ProductState):

    if state["quality_score"] >= 8:
        return "approved"

    return "needs_revision"


# 6. Node: Publish
def publish(state: ProductState):
    print("\n[PUBLISH] Description approved!")

    return {
        "status": "approved"
    }


# 7. Node: Needs revision
def needs_revision(state: ProductState):
    print("\n[REVISION] Description needs further improvement.")

    return {
        "status": "needs_revision"
    }


# 8. Create graph builder
builder = StateGraph(ProductState)


# 9. Register all nodes
builder.add_node("analyze", analyze)
builder.add_node("improve", improve)
builder.add_node("review", review)
builder.add_node("publish", publish)
builder.add_node("needs_revision", needs_revision)


# 10. Connect the initial workflow
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "improve")
builder.add_edge("improve", "review")


# 11. Add conditional routing
builder.add_conditional_edges(
    "review",
    route_review,
    {
        "approved": "publish",
        "needs_revision": "needs_revision"
    }
)


# 12. Connect both possible destinations to END
builder.add_edge("publish", END)
builder.add_edge("needs_revision", END)


# 13. Compile the graph
graph = builder.compile()


# 14. Execute the graph
initial_state = {
    "product_description": "This laptop is good and very fast.",
    "analysis": "",
    "improved_description": "",
    "quality_score": 0,
    "status": "pending"
}

result = graph.invoke(initial_state)


print("\n========== FINAL RESULT ==========")
print(result)