"""Tests for cross-platform path helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fiber_tracer.utils.paths import get_config_dir


def test_config_dir_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path))
    assert get_config_dir() == str(tmp_path)


def test_config_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "fiber-tracer"
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(target))
    assert get_config_dir() == str(target)
    assert target.exists()


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_config_dir_treats_empty_env_as_unset(value, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", value)
    result = get_config_dir()
    assert result != ""
    if os.name == "nt":
        expected = str(Path.home() / "AppData" / "Roaming" / "fiber-tracer")
    else:
        expected = str(Path.home() / ".config" / "fiber-tracer")
    assert result == expected
