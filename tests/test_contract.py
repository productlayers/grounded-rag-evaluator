import re
from pathlib import Path

import pytest

from src.generation.grounded_answer import generate_answer


def test_ui_backend_mode_sync():
    """Verify that modes listed in the UI match those accepted by the backend."""
    ui_path = Path("src/app/streamlit_app.py")
    if not ui_path.exists():
        pytest.skip("Streamlit app not found")

    ui_content = ui_path.read_text()

    # regex to find the radio button modes: ["mode1", "mode2"]
    # Looking for: ["retrieval", "llm"]
    match = re.search(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', ui_content)
    assert match, "Could not find mode list in streamlit_app.py"

    ui_modes = [match.group(1), match.group(2)]

    # Try calling the backend with each mode (mocked)
    for mode in ui_modes:
        try:
            # We don't care about the results, just that it doesn't raise
            # ValueError("Unknown generation mode")
            # We'll pass empty data to trigger insufficient evidence quickly
            generate_answer(question="test", mode=mode, min_score=1.0)
        except ValueError as e:
            if "Unknown generation mode" in str(e):
                pytest.fail(f"UI offers mode '{mode}' but Backend rejects it: {e}")
            raise e
        except Exception:
            # Other errors (like missing API keys) are fine for this contract test
            pass


def test_cli_backend_mode_sync():
    """Verify that modes in CLI args match logic."""
    main_path = Path("src/app/main.py")
    if not main_path.exists():
        pytest.skip("main.py not found")

    main_content = main_path.read_text()
    # Find choices=["...", "..."]
    matches = re.findall(r'choices=\["([^"]+)"\s*,\s*"([^"]+)"\]', main_content)

    for choices in matches:
        for mode in choices:
            try:
                generate_answer(question="test", mode=mode, min_score=1.0)
            except ValueError as e:
                if "Unknown generation mode" in str(e):
                    pytest.fail(f"CLI offers mode '{mode}' but Backend rejects it: {e}")
                raise e
            except Exception:
                pass
