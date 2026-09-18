"""Tests for TelemetryStorage database schema and indexes."""

from __future__ import annotations

from pathlib import Path

from redthread.config.settings import RedThreadSettings, TargetBackend
from redthread.telemetry.storage import TelemetryStorage


def test_telemetry_storage_creates_composite_indexes(tmp_path: Path) -> None:
    settings = RedThreadSettings(
        target_backend=TargetBackend.OLLAMA,
        target_model="llama3.2:3b",
        attacker_backend=TargetBackend.OLLAMA,
        attacker_model="llama3.2:3b",
        judge_backend=TargetBackend.OPENAI,
        judge_model="gpt-4o",
        openai_api_key="test-key",
        dry_run=True,
    ).model_copy(update={"data_dir": tmp_path})

    storage = TelemetryStorage(settings)

    with storage._connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_telemetry_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_telemetry_model_time" in indexes
        assert "idx_telemetry_canary" in indexes

        # Verify query plan uses the composite model_time index
        plan_cursor = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM telemetry_records WHERE target_model = 'gpt-4o' ORDER BY timestamp DESC"
        )
        plan_details = " ".join(row[3] for row in plan_cursor.fetchall())
        assert "idx_telemetry_model_time" in plan_details

        # Verify query plan uses the canary index
        canary_plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM telemetry_records WHERE is_canary = 1 AND canary_id = 'test-canary'"
        )
        canary_details = " ".join(row[3] for row in canary_plan.fetchall())
        assert "idx_telemetry_canary" in canary_details
