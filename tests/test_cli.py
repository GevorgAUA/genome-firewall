from __future__ import annotations

from typer.testing import CliRunner

from genome_firewall.cli import app


def test_cli_help_lists_demo_product_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "predict" in result.stdout
    assert "demo" in result.stdout
    assert "ui" in result.stdout
