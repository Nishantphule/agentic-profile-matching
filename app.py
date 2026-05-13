"""
app.py
======

Streamlit entrypoint.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence noisy module-introspection warnings BEFORE Streamlit imports.
#
# When Streamlit's local-sources watcher inspects every imported module, it
# triggers `transformers`' lazy attribute loader, which in turn tries to
# import vision-only sub-modules (e.g. SAM, YOLOS, Qwen2-VL) that depend on
# `torchvision`.  We don't need any of those for resume embeddings, but the
# resulting ModuleNotFoundError stack traces flood the console.
# ---------------------------------------------------------------------------
for _noisy in (
    "streamlit.watcher.local_sources_watcher",
    "streamlit.watcher",
    "transformers",
    "transformers.modeling_utils",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", message=r".*Examining the path of.*")

# Make sure absolute imports work whether you run `streamlit run app.py`
# from inside the project folder or via `python -m`.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ui.chat_interface import render  # noqa: E402  (after sys.path mutation)


if __name__ == "__main__":
    render()
else:
    # Streamlit runs the script top-to-bottom each rerun, so make sure
    # `render()` is invoked regardless of `__main__`.
    render()
