import pytest

from graph_modi.models.soft_prompt import SoftPromptConfig


def test_soft_prompt_config_defaults_match_readme_spec() -> None:
    config = SoftPromptConfig()
    assert config.prefix_tokens == 10
    assert "{question}" in config.prompt_template


def test_soft_prompt_config_rejects_non_positive_prefix_tokens() -> None:
    with pytest.raises(ValueError, match="prefix_tokens"):
        SoftPromptConfig(prefix_tokens=0)


def test_soft_prompt_config_rejects_non_positive_max_sequence_length() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        SoftPromptConfig(max_sequence_length=0)


def test_soft_prompt_config_requires_question_placeholder() -> None:
    with pytest.raises(ValueError, match="prompt_template"):
        SoftPromptConfig(prompt_template="Answer this:")
