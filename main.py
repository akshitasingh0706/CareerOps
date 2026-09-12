from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


load_dotenv()  # Load environment variables from .env file

from graph import graph

console = Console()

jd_folder = Path("jds")

for jd_file in jd_folder.glob("*.txt"):
    jd_text = jd_file.read_text(encoding="utf-8")
    
    result = graph.invoke({"job_text": jd_text})

    analysis = result["analysis"]

    console.print()
    console.print(
        Panel(
            f"[bold cyan]{analysis.role_title}[/bold cyan]\n"
            f"[yellow]{analysis.seniority_level}[/yellow]",
            title=f"📄 {jd_file.name}",
            border_style="cyan"
        )
    )

    # Key requirements
    console.print("\n[bold]Key Requirements[/bold]")

    requirements = Table(show_header=False, box=None)
    requirements.add_column("Requirement")

    for requirement in analysis.key_requirements:
        requirements.add_row(f"• {requirement}")

    console.print(requirements)

    # Fit summary
    console.print(
        Panel(
            analysis.fit_summary,
            title="🎯 Fit Summary",
            border_style="green"
        )
    )

    # Gaps
    console.print("\n[bold red]Potential Gaps[/bold red]")

    for gap in analysis.gaps:
        console.print(f"  • {gap}")

    console.print()
    
