from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from redthread.cli import main
from redthread.reporting import (
    build_operator_artifact_bundle,
    operator_artifacts_to_sarif,
    operator_artifacts_to_sarif_dict,
    write_operator_artifacts,
)
from tests.operator_reporting_helpers import make_campaign


def test_sarif_dict_structure_and_rules() -> None:
    campaign = make_campaign()
    bundle = build_operator_artifact_bundle(campaign)
    sarif = operator_artifacts_to_sarif_dict(bundle)

    assert sarif["version"] == "2.1.0"
    assert "sarif-schema-2.1.0.json" in sarif["$schema"]
    assert len(sarif["runs"]) == 1

    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "RedThread"
    assert len(run["tool"]["driver"]["rules"]) >= 1

    results = run["results"]
    assert len(results) == 1
    res = results[0]
    assert res["level"] in ("error", "warning", "note")
    assert "Finding" in res["message"]["text"] or "Confirmed vulnerability" in res["message"]["text"]
    assert len(res["locations"]) == 1
    assert "artifactLocation" in res["locations"][0]["physicalLocation"]
    assert "properties" in res
    assert "judgeScore" in res["properties"]


def test_write_operator_artifacts_with_sarif(tmp_path: Path) -> None:
    campaign = make_campaign()
    bundle = build_operator_artifact_bundle(campaign)
    sarif_file = tmp_path / "findings.sarif"

    write_operator_artifacts(bundle, sarif_path=sarif_file)

    assert sarif_file.exists()
    payload = json.loads(sarif_file.read_text(encoding="utf-8"))
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["tool"]["driver"]["name"] == "RedThread"


def test_operator_artifacts_to_sarif_serialized_string() -> None:
    campaign = make_campaign()
    bundle = build_operator_artifact_bundle(campaign)
    sarif_str = operator_artifacts_to_sarif(bundle)

    data = json.loads(sarif_str)
    assert data["version"] == "2.1.0"
    assert isinstance(data["runs"], list)


def test_cli_report_sarif_option(tmp_path: Path) -> None:
    sarif_output = tmp_path / "results.sarif"
    campaign = make_campaign()

    with patch("redthread.cli.run.RedThreadEngine.run") as mock_engine_run:
        mock_engine_run.return_value = campaign
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--dry-run",
                "--report-sarif",
                str(sarif_output),
            ],
        )

        assert result.exit_code == 1
        assert sarif_output.exists()
        loaded = json.loads(sarif_output.read_text(encoding="utf-8"))
        assert loaded["version"] == "2.1.0"
