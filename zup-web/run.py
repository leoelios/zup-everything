"""Quick-start: python run.py"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765", "--reload"],
    cwd=__file__.replace("run.py", ""),
)
