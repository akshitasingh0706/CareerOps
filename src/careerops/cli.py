"""Command line interface. Run `careerops --help`."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from careerops import service
from careerops.domain import Stage

app = typer.Typer(help="CareerOps: run your job search.", no_args_is_help=True)
console = Console()


@app.command()
def watch(ats: str, token: str, name: str = typer.Option(None)) -> None:
    """Watch a company board. ATS is greenhouse, lever or ashby."""
    company = service.watch_company(ats, token, name)
    console.print(f"Watching {company.name} ({company.id})")


@app.command()
def poll() -> None:
    """Fetch jobs from every watched company."""
    new = service.poll()
    console.print(f"{len(new)} new jobs")


@app.command()
def jobs(limit: int = 25) -> None:
    """Show jobs ranked for your profile."""
    result = service.rank(limit)
    table = Table("score", "title", "company", "location", "id")
    for r in result.ranked:
        table.add_row(
            f"{r.final_score:.2f}", r.job.title, r.job.company_name, r.job.location or "-", r.job.id[:10]
        )
    console.print(table)
    console.print(f"personalization: {result.personalization.status}")
    if result.filtered_out:
        console.print(f"{len(result.filtered_out)} jobs failed a hard filter")


@app.command()
def save(job_id: str) -> None:
    """Save a job to your pipeline. Use the full job id."""
    app_ = service.save_job(job_id)
    console.print(f"Saved. Application {app_.id}, stage {app_.stage.value}")


@app.command()
def skip(job_id: str) -> None:
    """Hide a job and teach the ranker you don't want it."""
    service.skip_job(job_id)
    console.print("Skipped")


@app.command()
def advance(app_id: str, stage: Stage) -> None:
    """Move an application to a new stage."""
    result = service.advance(app_id, stage)
    console.print(f"{result.id} is now {result.stage.value}")


@app.command()
def pipeline() -> None:
    """Applications per stage."""
    for stage, n in service.pipeline().items():
        console.print(f"{stage:12} {n}")


@app.command()
def due() -> None:
    """Follow-ups and deadlines due today or earlier."""
    items = service.due_followups()
    for d in items:
        console.print(f"{d['due_on']}  {d['kind']:9} {d['note']}")
    if not items:
        console.print("Nothing due")


if __name__ == "__main__":
    app()
