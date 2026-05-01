"""BioMedArena — A State-of-the-Art Biomedical Harness for Evaluating AI Agents at Scale."""

# Bootstrap MUST run before any module that calls os.environ / config loaders
from harness._bootstrap import load_env_once as _load_env_once

_load_env_once()
