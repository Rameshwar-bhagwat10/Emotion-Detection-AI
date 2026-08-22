"""Tests for Alembic schema migrations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_alembic_migration_upgrade_and_downgrade() -> None:
    """Verify that Alembic migrations successfully apply to a clean database and downgrade cleanly."""
    root_dir = Path(__file__).resolve().parent.parent
    alembic_ini_path = root_dir / "alembic.ini"
    assert alembic_ini_path.exists(), f"alembic.ini not found at {alembic_ini_path}"

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        tmp_db_path = tmp_db.name

    try:
        sqlite_url = f"sqlite:///{tmp_db_path.replace(os.sep, '/')}"

        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", sqlite_url)
        alembic_cfg.set_main_option("script_location", str(root_dir / "alembic"))

        # 1. Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # 2. Inspect created tables
        sync_engine = create_engine(sqlite_url)
        inspector = inspect(sync_engine)
        tables = set(inspector.get_table_names())

        expected_tables = {
            "users",
            "sessions",
            "predictions",
            "face_predictions",
            "model_versions",
            "alembic_version",
        }
        assert expected_tables.issubset(
            tables
        ), f"Missing tables in migration: {expected_tables - tables}"

        # Verify columns on predictions table
        pred_columns = {c["name"] for c in inspector.get_columns("predictions")}
        assert {
            "id",
            "session_id",
            "request_id",
            "model_version",
            "status",
            "faces_detected",
            "processing_time_ms",
        }.issubset(pred_columns)

        # Verify columns on face_predictions table
        face_columns = {c["name"] for c in inspector.get_columns("face_predictions")}
        assert {
            "id",
            "prediction_id",
            "face_id",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "detection_confidence",
            "emotion",
            "confidence",
            "probabilities",
        }.issubset(face_columns)

        sync_engine.dispose()

        # 3. Downgrade to base
        command.downgrade(alembic_cfg, "base")

        sync_engine_down = create_engine(sqlite_url)
        inspector_down = inspect(sync_engine_down)
        remaining_tables = set(inspector_down.get_table_names())
        assert "predictions" not in remaining_tables
        assert "face_predictions" not in remaining_tables
        assert "sessions" not in remaining_tables
        assert "users" not in remaining_tables
        sync_engine_down.dispose()

        # 4. Upgrade back to head
        command.upgrade(alembic_cfg, "head")

    finally:
        if os.path.exists(tmp_db_path):
            try:
                os.remove(tmp_db_path)
            except PermissionError:
                pass
