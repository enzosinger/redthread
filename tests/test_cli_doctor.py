from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from click.testing import CliRunner
from rich.console import Console

from redthread.cli import main
from redthread.cli.doctor import run_doctor
from redthread.cli.doctor_checks import DoctorCheck, collect_doctor_checks
from redthread.config.settings import RedThreadSettings, TargetBackend


def test_doctor_in_dry_run_reports_info_for_missing_endpoints(tmp_path: Path) -> None:
    settings = RedThreadSettings(
        dry_run=True,
        log_dir=tmp_path / "logs",
        memory_dir=tmp_path / "memory",
        target_backend=TargetBackend.OLLAMA,
        attacker_backend=TargetBackend.OLLAMA,
        openai_api_key="",
    )

    with patch("redthread.cli.doctor_checks.urlopen", side_effect=URLError("Connection refused")):
        checks = collect_doctor_checks(settings, env_file=str(tmp_path / ".env"))

        ollama_checks = [c for c in checks if c.name == "Ollama reachability"]
        assert len(ollama_checks) == 1
        assert ollama_checks[0].status == "info"
        assert "offline/dry-run mode active" in ollama_checks[0].detail

        openai_checks = [c for c in checks if c.name == "OpenAI key"]
        assert len(openai_checks) == 1
        assert openai_checks[0].status == "info"


def test_doctor_in_live_mode_warns_with_remediation(tmp_path: Path) -> None:
    settings = RedThreadSettings(
        dry_run=False,
        log_dir=tmp_path / "logs",
        memory_dir=tmp_path / "memory",
        target_backend=TargetBackend.OLLAMA,
        attacker_backend=TargetBackend.OLLAMA,
    )

    with patch("redthread.cli.doctor_checks.urlopen", side_effect=URLError("Connection refused")):
        checks = collect_doctor_checks(settings, env_file=str(tmp_path / ".env"))

        ollama_checks = [c for c in checks if c.name == "Ollama reachability"]
        assert len(ollama_checks) == 1
        assert ollama_checks[0].status == "warn"
        assert "ollama serve" in ollama_checks[0].detail


def test_doctor_run_returns_exit_code_zero_when_no_fails(tmp_path: Path) -> None:
    settings = RedThreadSettings(
        dry_run=True,
        log_dir=tmp_path / "logs",
        memory_dir=tmp_path / "memory",
    )
    console = Console(record=True)
    exit_code = run_doctor(console, settings, env_file=str(tmp_path / ".env"))
    assert exit_code == 0


def test_doctor_run_returns_exit_code_one_when_fail(tmp_path: Path) -> None:
    settings = RedThreadSettings(
        dry_run=True,
        log_dir=tmp_path / "logs",
        memory_dir=tmp_path / "memory",
    )
    fake_checks = [
        DoctorCheck("System", "fail", "Disk failure"),
    ]
    with patch("redthread.cli.doctor.collect_doctor_checks", return_value=fake_checks):
        console = Console(record=True)
        exit_code = run_doctor(console, settings, env_file=str(tmp_path / ".env"))
        assert exit_code == 1


def test_doctor_cli_command(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor", "--env-file", str(tmp_path / ".env")])
    assert result.exit_code == 0
    assert "REDTHREAD DOCTOR" in result.output
