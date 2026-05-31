import sys
from pathlib import Path

# Make the repo-root asr_bench.py importable regardless of pytest's CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
