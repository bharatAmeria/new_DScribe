"""CLI entry point for the Discharge Summary Agent."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

app = typer.Typer(help="Discharge Summary Agent — Agentic AI for clinical discharge summaries")
console = Console()


@app.command()
def run(
    patient_folder: str = typer.Argument(..., help="Path to patient folder containing PDFs"),
    patient_id: str = typer.Option("", "--id", help="Patient ID (defaults to folder name)"),
    output_dir: str = typer.Option("outputs", "--output", "-o", help="Output directory"),
    max_iterations: int = typer.Option(20, "--max-iter", help="Max agent iterations (safety cap)"),
    model: str = typer.Option("", "--model", help="Override LLM model"),
):
    """Run the discharge summary agent on a patient folder."""
    # Validate
    folder = Path(patient_folder)
    if not folder.exists():
        console.print(f"[red]Error: folder not found: {folder}[/red]")
        raise typer.Exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[red]Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        raise typer.Exit(1)

    if model:
        os.environ["DISCHARGE_MODEL"] = model

    patient_id = patient_id or folder.name

    console.print(Panel(
        f"[bold]Patient:[/bold] {patient_id}\n"
        f"[bold]Folder:[/bold] {folder}\n"
        f"[bold]Output:[/bold] {output_dir}\n"
        f"[bold]Max iterations:[/bold] {max_iterations}",
        title="[cyan]Discharge Summary Agent[/cyan]",
        border_style="cyan",
    ))

    from .graph import run_agent
    result = run_agent(
        patient_id=patient_id,
        patient_folder=folder,
        output_dir=output_dir,
        max_iterations=max_iterations,
    )

    # Print summary
    summary = result.get("summary")
    if summary:
        table = Table(title="Summary Overview", border_style="green")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Patient", summary.patient_name or "[MISSING]")
        table.add_row("Admission", summary.admission_date or "[MISSING]")
        table.add_row("Discharge", summary.discharge_date or "[MISSING]")
        table.add_row("Principal Dx", summary.principal_diagnosis or "[MISSING]")
        table.add_row("Condition", summary.discharge_condition or "[MISSING]")
        table.add_row("Flags", str(len(summary.flags)))
        table.add_row("Missing Fields", str(len(summary.missing_fields)))
        table.add_row("Escalations", str(len(result.get("escalation_log", []))))
        console.print(table)

    console.print(f"\n[green]✓ Trace saved:[/green] {result['trace_path']}")
    console.print(f"[green]✓ Summary saved in:[/green] {output_dir}/")

    if result.get("errors"):
        console.print(f"[yellow]Errors:[/yellow] {result['errors']}")


@app.command()
def batch(
    patients_dir: str = typer.Argument(..., help="Directory containing patient subfolders"),
    output_dir: str = typer.Option("outputs", "--output", "-o"),
    max_iterations: int = typer.Option(20, "--max-iter"),
):
    """Run the agent on all patients in a directory."""
    base = Path(patients_dir)
    patient_folders = [d for d in sorted(base.iterdir()) if d.is_dir()]

    if not patient_folders:
        console.print(f"[red]No patient folders found in {base}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Running on {len(patient_folders)} patients...[/cyan]")
    for folder in patient_folders:
        console.rule(f"Patient: {folder.name}")
        from .graph import run_agent
        try:
            run_agent(
                patient_id=folder.name,
                patient_folder=folder,
                output_dir=output_dir,
                max_iterations=max_iterations,
            )
        except Exception as e:
            console.print(f"[red]Failed: {e}[/red]")


if __name__ == "__main__":
    app()
