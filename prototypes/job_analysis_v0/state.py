from typing import TypedDict, Optional
from models import JobAnalysis

class JobAnalysisState(TypedDict):
    job_text: str
    analysis: Optional[JobAnalysis]

    tools: list[str]
    tool_results: dict

    final_recommendation: str | None

    iterations: int
    max_iterations: int
    max_iterations_reached: bool

