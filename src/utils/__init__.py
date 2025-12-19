# Utils package
# Re-export functions from utils.py for backward compatibility
import sys
import importlib.util
from pathlib import Path

# Import from the parent utils.py file
_parent_dir = Path(__file__).parent.parent
_utils_file = _parent_dir / "utils.py"

if _utils_file.exists():
    # Load utils.py as a module
    spec = importlib.util.spec_from_file_location("_utils_py_module", _utils_file)
    _utils_py = importlib.util.module_from_spec(spec)
    sys.modules['_utils_py_module'] = _utils_py
    spec.loader.exec_module(_utils_py)

    # Re-export the functions
    calculate_length_ratio = _utils_py.calculate_length_ratio
    should_skip_length_gate = _utils_py.should_skip_length_gate
    is_very_different_length = _utils_py.is_very_different_length
    is_moderate_different_length = _utils_py.is_moderate_different_length
    get_length_gate_ratios = _utils_py.get_length_gate_ratios
else:
    raise ImportError(f"Could not find utils.py at {_utils_file}")
