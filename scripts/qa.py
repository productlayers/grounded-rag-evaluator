#!/usr/bin/env python3
import subprocess
import sys


def run_step(name, command):
    print(f"\n>>> Running {name}: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"ok: {name} passed.")
        return True
    except subprocess.CalledProcessError:
        print(f"FAIL: {name} failed.")
        return False


def main():
    steps = [
        ("Format (Ruff)", ["ruff", "format", "."]),
        ("Lint (Ruff)", ["ruff", "check", "--fix", "."]),
        ("Contract Tests", ["pytest", "tests/test_contract.py", "-v"]),
        ("Logic Tests", ["pytest", "tests/test_generation.py", "tests/test_retrieval.py", "-v"]),
    ]

    success = True
    for name, cmd in steps:
        if not run_step(name, cmd):
            success = False

    if success:
        print("\n✅ QA Loop Passed! Ready to ship.")
    else:
        print("\n❌ QA Loop Failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
