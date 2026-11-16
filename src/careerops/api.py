"""HTTP API for the web app. Run with: uvicorn careerops.api:app --reload"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from careerops import service
from careerops.crm.service import CRMError
from careerops.domain import Stage, UserProfile

app = FastAPI(title="CareerOps")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CAREEROPS_WEB_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class WatchIn(BaseModel):
    ats: str  # greenhouse, lever or ashby
    token: str
    name: str | None = None


class AdvanceIn(BaseModel):
    stage: Stage
    note: str = ""


class SkipIn(BaseModel):
    reason: str | None = None


def _ranked_view(r) -> dict:
    j = r.job
    return {
        "id": j.id,
        "title": j.title,
        "company": j.company_name,
        "location": j.location,
        "remote": j.remote,
        "url": j.url,
        "source": j.source.value,
        "seniority": j.seniority.value,
        "salary_min": j.salary_min,
        "salary_max": j.salary_max,
        "score": round(r.final_score, 3),
        "components": {k: round(v, 3) for k, v in r.fit.components.items()},
        "unknown_filters": r.unknown_filters,
        "failed_filters": r.failed_filters,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/profile")
def get_profile() -> UserProfile:
    return service.load_profile()


@app.put("/profile")
def put_profile(profile: UserProfile) -> UserProfile:
    return service.save_profile(profile)


@app.get("/companies")
def get_companies():
    return service.companies()


@app.post("/companies")
def post_company(body: WatchIn):
    try:
        return service.watch_company(body.ats, body.token, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/poll")
def post_poll() -> dict:
    new = service.poll()
    return {"new_jobs": len(new)}


@app.get("/jobs/ranked")
def ranked(limit: int = 50) -> dict:
    result = service.rank(limit)
    return {
        "ranked": [_ranked_view(r) for r in result.ranked],
        "filtered_out": [_ranked_view(r) for r in result.filtered_out[:limit]],
        "personalization": result.personalization.model_dump(mode="json"),
    }


@app.post("/jobs/{job_id}/save")
def save_job(job_id: str):
    try:
        return service.save_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc


@app.post("/jobs/{job_id}/skip")
def skip_job(job_id: str, body: SkipIn | None = None) -> dict:
    try:
        service.skip_job(job_id, body.reason if body else None)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    return {"ok": True}


@app.get("/applications")
def get_applications():
    return service.applications()


@app.post("/applications/{app_id}/advance")
def advance(app_id: str, body: AdvanceIn):
    try:
        return service.advance(app_id, body.stage, body.note)
    except CRMError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/pipeline")
def get_pipeline() -> dict:
    return service.pipeline()


@app.get("/followups/due")
def followups_due():
    return service.due_followups()
