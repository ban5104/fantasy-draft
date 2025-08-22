#!/usr/bin/env python3
"""
Standard test script for the fantasy draft optimizer.
Runs comprehensive optimization with quality checks and data export.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_optimizer_test():
    """Run the standard optimizer test with consistent parameters."""

    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Standard test command with consistent parameters
    cmd = [
        "python3",
        "scripts/dp_draft_optimizer_debug.py",
        "--sims",
        "10000",
        "--export-csv",
    ]

    print("Running standard draft optimizer test...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("\nExported files available in data/output-simulations/:")
        print("  - mc_player_survivals.csv")
        print("  - mc_position_summary.csv")
        print("  - mc_config.csv")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = run_optimizer_test()
    sys.exit(0 if success else 1)
