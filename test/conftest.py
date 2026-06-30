"""Pytest configuration for GS tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU and CuPy")
