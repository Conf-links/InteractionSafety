# Environment shared by every script in this folder.
# PY: the Python interpreter of the environment built from ../../../requirements.txt (override with `PY=... bash ...`).
export PY=${PY:-python}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3
# Optional, machine-specific:
#   export HF_HUB_OFFLINE=1        # once all models are in the Hugging Face cache
#   export LD_LIBRARY_PATH=...     # only if torch reports "CUDA Error 803" because of a CUDA compat library
