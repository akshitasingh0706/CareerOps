from langchain_anthropic import ChatAnthropic
from state import JobAnalysisState
from models import JobAnalysis

llm = ChatAnthropic(
    model = "claude-haiku-4-5-20251001",
    temperature = 0,
)

structured_llm = llm.with_structured_output(JobAnalysis)

def analyze_job(
    state: JobAnalysisState
    ):

    jd = state['job_text'][:6000]

    prompt = f"""
    Analyze the following job description.

    Return a JSON object with the following fields:
    - role_title: The title of the role.
    - seniority_level: The seniority level of the role (e.g., Junior, Mid-level, Senior).
    - key_requirements: A list of key requirements for the role.
    - fit_summary: A brief summary of how well a candidate with the given job description would fit the role.
    - gaps: A list of any gaps or missing qualifications for the role.

    Job Description:
    {jd}
    """

    analysis = structured_llm.invoke(prompt)

    return {
        "analysis": analysis
    }