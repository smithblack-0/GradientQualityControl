"""
When run, this small python script will attempt to
fix any style issues by running black and then
ruff back-to-back
"""

# !/usr/bin/env python
"""
Run code formatters and linters.
Usage: python format_code.py
"""

import subprocess
import sys


def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    success = True

    # Format with Black
    if not run_command(["black", "src/", "tests/"]):
        print("Black formatting failed")
        success = False

    # Fix issues with Ruff
    if not run_command(["ruff", "check", "--fix", "src/", "tests/"]):
        print("Ruff fixes failed")
        success = False

    if success:
        print("\nAll formatting complete. Review changes and commit.")
    else:
        print("\nSome issues encountered. Check output above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())