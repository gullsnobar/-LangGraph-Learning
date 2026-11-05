from langgraph.graph import StateGraph, START, END


# Node 1
def analyze(state):
    print("Step 1: Analyzing content")
    return state


# Node 2
def improve(state):
    print("Step 2: Improving content")
    return state


# Node 3
def review(state):
    print("Step 3: Reviewing content")
    return state


# Create a graph builder
builder = StateGraph(dict)


# Register nodes
builder.add_node("analyze", analyze)
builder.add_node("improve", improve)
builder.add_node("review", review)


# Connect nodes
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "improve")
builder.add_edge("improve", "review")
builder.add_edge("review", END)


# Compile the graph
graph = builder.compile()


# Execute the graph
result = graph.invoke({})

print("\nGraph execution completed!")
print("Final state:", result)