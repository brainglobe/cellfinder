"""Smoke tests ensuring CLI --help option exits without failure"""

import subprocess

import pytest


@pytest.mark.parametrize(
    "entry_point",
    [
        "cellfinder_download",
        "cellfinder_train",
        "cellfinder",
    ],
)
def test_cli_help(entry_point):
    result = subprocess.run(
        [entry_point, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`{entry_point} --help` exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
