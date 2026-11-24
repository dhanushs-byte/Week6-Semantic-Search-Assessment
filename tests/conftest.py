
import os
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enable lightweight test mode
os.environ["WEEK6_TEST_MODE"] = "1"
os.environ["API_KEY"] = "test-secret"
