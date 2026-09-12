from langgraph.graph import StateGraph, START, END
from state import JobAnalysisState
from nodes import analyze_job

builder = StateGraph(JobAnalysisState)

builder.add_node("analyze_job", analyze_job)

builder.add_edge(START, "analyze_job")
builder.add_edge("analyze_job", END)

graph = builder.compile()

result = graph.invoke({
    "job_text": "We are looking for a Software Engineer with experience in Python and Django. The ideal candidate should have a strong understanding of REST APIs and be able to work in a collaborative environment."
})

print(result)
