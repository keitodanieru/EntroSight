"""Script to run property tests and print results."""
import sys
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_heatmap_properties.py", "-v"],
    capture_output=True,
    text=True,
    timeout=300,
)
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print("RETURN CODE:", result.returncode)
