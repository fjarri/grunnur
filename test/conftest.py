import pytest

from grunnur.pytest_helpers import *


def pytest_addoption(parser):
    addoption(parser)


def pytest_generate_tests(metafunc):
    generate_tests(metafunc)


def pytest_report_header(config):
    report_header(config)
