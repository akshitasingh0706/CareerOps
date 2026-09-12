from typing import TypedDict, Optional
from models import JobAnalysis

class JobAnalysisState(TypedDict):
    job_text: str
    analysis: Optional[JobAnalysis]
