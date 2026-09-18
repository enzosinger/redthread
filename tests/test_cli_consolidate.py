from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from redthread.cli import main
from redthread.memory.consolidation import ConsolidationReport


def test_consolidate_command_with_empty_logs(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path(".env").write_text("REDTHREAD_DRY_RUN=true\n", encoding="utf-8")
        Path("logs").mkdir()
        Path("memory").mkdir()

        result = runner.invoke(main, ["consolidate"])

        assert result.exit_code == 0
        assert "Dream Memory Consolidation" in result.output
        assert "Scanned Log Files" in result.output
        assert "Memory consolidation complete." in result.output


def test_consolidate_command_with_errors() -> None:
    runner = CliRunner()
    fake_report = ConsolidationReport(
        scanned_files=1,
        new_jailbreaks_found=1,
        defenses_synthesized=0,
        defenses_validated=0,
        skipped_duplicates=0,
        errors=["Failed to synthesize defense: timeout"],
    )

    with patch("redthread.cli.consolidate.DreamConsolidator.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = fake_report
        result = runner.invoke(main, ["consolidate"])

        assert result.exit_code == 0
        assert "Dream Memory Consolidation" in result.output
        assert "Warning:" in result.output
        assert "Failed to synthesize defense: timeout" in result.output
