"""Tests for scripts/download_gfpa66.py.

These tests mock the Zenodo API and do not perform any network I/O.
"""

from unittest.mock import MagicMock, patch

import pytest

import scripts.download_gfpa66 as download_gfpa66


@pytest.fixture
def fake_metadata():
    return {
        "files": [
            {
                "key": "pa66_volumes.h5",
                "size": 42,
                "links": {"self": "https://zenodo.org/api/files/fake/pa66_volumes.h5"},
            },
            {
                "key": "readme.txt",
                "size": 12,
                "links": {"self": "https://zenodo.org/api/files/fake/readme.txt"},
            },
        ]
    }


def test_list_files_returns_expected_structure(fake_metadata):
    files = download_gfpa66.list_files(fake_metadata)
    assert len(files) == 2
    assert files[0] == {
        "key": "pa66_volumes.h5",
        "size": 42,
        "url": "https://zenodo.org/api/files/fake/pa66_volumes.h5",
    }


def test_download_with_accept_license_writes_file(tmp_path, fake_metadata):
    dest = tmp_path / "pa66_volumes.h5"

    def fake_get(url, **kwargs):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.headers = {"content-length": "42"}
        mock.iter_content = MagicMock(return_value=[b"a" * 21, b"b" * 21])
        return mock

    with (
        patch.object(download_gfpa66, "fetch_record_metadata", return_value=fake_metadata),
        patch.object(download_gfpa66.requests, "get", side_effect=fake_get),
    ):
        exit_code = download_gfpa66.main(
            [
                "--output-dir",
                str(tmp_path),
                "--file",
                "pa66_volumes.h5",
                "--accept-license",
            ]
        )

    assert exit_code == 0
    assert dest.exists()
    assert dest.read_bytes() == b"a" * 21 + b"b" * 21


def test_download_without_accept_license_returns_one(tmp_path, fake_metadata):
    with patch.object(download_gfpa66, "fetch_record_metadata", return_value=fake_metadata):
        exit_code = download_gfpa66.main(
            [
                "--output-dir",
                str(tmp_path),
                "--file",
                "pa66_volumes.h5",
            ]
        )
    assert exit_code == 1


def test_list_prints_file_names(capsys, fake_metadata):
    with patch.object(download_gfpa66, "fetch_record_metadata", return_value=fake_metadata):
        exit_code = download_gfpa66.main(["--list"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "pa66_volumes.h5" in captured
    assert "readme.txt" in captured
