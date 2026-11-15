import time

import pytest

from clinicalchat.llm.copilot_cli import CopilotCLIChatModel, CopilotCLIError, is_rate_limited


def test_rate_limit_detection():
    assert is_rate_limited("429 Sorry, you've exceeded your rate limit")
    assert not is_rate_limited("connection reset")


def test_cooldown_blocks_calls_without_spawning_cli(tmp_path, monkeypatch):
    m = CopilotCLIChatModel(model="gpt-4.1", copilot_home=tmp_path)
    monkeypatch.setattr(CopilotCLIChatModel, "_cooldown_until", time.time() + 60)
    with pytest.raises(CopilotCLIError, match="cool-down"):
        m._check_cooldown()
