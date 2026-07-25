from __future__ import annotations

import importlib


def test_dashboard_import():
    dashboard_module = importlib.import_module("alphagrid.dashboard.app")
    assert dashboard_module is not None
