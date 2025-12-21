"""Helper functions for tests to avoid skipping."""
from pathlib import Path
import json
from unittest.mock import Mock, patch


def ensure_config_exists():
    """Ensure config.json exists with minimal configuration for tests."""
    config_path = Path("config.json")
    if not config_path.exists():
        minimal_config = {
            "provider": "deepseek",
            "deepseek": {
                "api_key": "test-key",
                "api_url": "https://api.deepseek.com/v1/chat/completions",
                "model": "deepseek-chat"
            },
            "critic": {"fallback_pass_threshold": 0.75},
            "blend": {"authors": ["TestAuthor"]}
        }
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)
    else:
        # Ensure existing config has required fields (for CI)
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Add missing fields if they don't exist
        updated = False
        if "deepseek" not in config:
            config["deepseek"] = {}
            updated = True

        deepseek = config["deepseek"]
        if "api_key" not in deepseek or not deepseek.get("api_key"):
            deepseek["api_key"] = "test-key"
            updated = True
        if "api_url" not in deepseek or not deepseek.get("api_url"):
            deepseek["api_url"] = "https://api.deepseek.com/v1/chat/completions"
            updated = True

        if updated:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    return config_path


def mock_llm_provider_for_critic():
    """Create a mock LLM provider for critic_evaluate tests."""
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps({
        "pass": True,
        "score": 0.85,
        "feedback": "Text matches structure and situation well."
    })
    mock_llm_class = Mock(return_value=mock_llm)
    return patch('src.validator.critic.LLMProvider', mock_llm_class)

