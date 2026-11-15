"""ClinicalChat — simple role-aware multi-agent clinical chatbot."""

__version__ = "0.1.0"

import os as _os

_os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
_os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb
