from pydantic import BaseModel

class JobAnalysis(BaseModel):
    role_title: str
    seniority_level: str
    key_requirements: list[str]
    fit_summary: str
    gaps: list[str]
