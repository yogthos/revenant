import pytest


@pytest.fixture(autouse=True)
def no_deepseek_key(monkeypatch):
    """Unit tests never call DeepSeek: without a key call_deepseek raises."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
