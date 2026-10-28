"""Acceptance-test harness.

Tests 1-3 judge an actual run: set MIE27_RUN_DIR=runs/<run_id>. On finish, the
outcome of every test is written to <run_dir>/acceptance.json; the figure script
only builds final (non-DRAFT) figures when all tests passed and none were skipped.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

TOL_OPT = 0.01
REQUIRED_REPS = 200
_outcomes: dict[str, str] = {}


def run_dir():
    d = os.environ.get("MIE27_RUN_DIR")
    return Path(d) if d else None


@pytest.fixture(scope="session")
def results():
    d = run_dir()
    if d is None or not (d / "results.csv").exists():
        pytest.skip("set MIE27_RUN_DIR to a run with results.csv")
    return pd.read_csv(d / "results.csv")


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _outcomes[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    d = run_dir()
    if d is None or not d.exists():
        return
    ok = bool(_outcomes) and all(v == "passed" for v in _outcomes.values())
    (d / "acceptance.json").write_text(json.dumps(dict(
        all_passed=ok, when=datetime.now(timezone.utc).isoformat(),
        outcomes=_outcomes), indent=1))
