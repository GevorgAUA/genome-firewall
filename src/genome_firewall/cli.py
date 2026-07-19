"""Convenient Genome Firewall commands for inference and the hackathon demo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from genome_firewall.config import load_config
from genome_firewall.predict import predict_cohort_sample, predict_genome

app = typer.Typer(
    no_args_is_help=True,
    help="Interpretable, target-gated E. coli AMR prediction research prototype.",
)


@app.command("predict")
def predict_command(
    fasta: Annotated[Path, typer.Option("--fasta", help="Assembled nucleotide FASTA")],
    antibiotic: Annotated[
        str | None,
        typer.Option("--antibiotic", help="ciprofloxacin or ampicillin"),
    ] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/mvp.yaml"),
) -> None:
    """Run AMRFinder and frozen predictions for a FASTA."""

    result = predict_genome(fasta, antibiotic, config_path=config)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("demo")
def demo_command(
    sample_id: Annotated[str, typer.Option("--sample-id")] = "562.100008",
    antibiotic: Annotated[str | None, typer.Option("--antibiotic")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/mvp.yaml"),
) -> None:
    """Return an instant prediction using cached cohort reports."""

    loaded = load_config(config)
    result = predict_cohort_sample(sample_id, loaded, antibiotic)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("ui")
def ui_command() -> None:
    """Launch the local Streamlit product interface."""

    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(project_root / "app.py")],
        check=True,
        cwd=project_root,
    )


if __name__ == "__main__":
    app()
