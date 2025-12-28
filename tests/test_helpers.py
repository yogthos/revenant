"""Test helpers and utilities."""

import shutil
from pathlib import Path


def ensure_config_exists():
    """Ensure config.json exists by copying from sample if needed."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.json"
    sample_path = project_root / "config.json.sample"

    if not config_path.exists() and sample_path.exists():
        shutil.copy(sample_path, config_path)
        print(f"Created {config_path} from {sample_path}")
    elif config_path.exists():
        print(f"Config already exists: {config_path}")
    else:
        print(f"Warning: No config.json.sample found at {sample_path}")
