import pytest

from grunnur import API, Context, CUDA_API_ID
from grunnur.api import all_api_ids
from grunnur.pytest_helpers import *

from .utils import mock_backend


@pytest.fixture
def mock_context(request, monkeypatch):
    api_id = request.param
    mock_backend(monkeypatch, api_id, [('Platform1', ['Device1'])])
    api = API.from_api_id(api_id)
    context = Context.from_criteria(api)
    yield context

    # The yielded object is preserved somewhere inside PyTest, so there is a race condition
    # between `context` destructor and `monkeypatch` rollback, which leads to
    # actual PyCUDA context being popped instead of the mock.
    # So we are releasing the context stack manually in the PyCUDA case.

    if api_id == CUDA_API_ID:
        context._context_adapter._context_stack = None

def generate_mock_tests(metafunc):
    if 'mock_context' in metafunc.fixturenames:
        api_ids = all_api_ids()
        test_ids = [f"mock-{api_id.shortcut}-context" for api_id in api_ids]
        metafunc.parametrize('mock_context', api_ids, ids=test_ids, indirect=True)


def pytest_addoption(parser):
    addoption(parser)


def pytest_generate_tests(metafunc):
    generate_tests(metafunc)
    generate_mock_tests(metafunc)


def pytest_report_header(config):
    report_header(config)
