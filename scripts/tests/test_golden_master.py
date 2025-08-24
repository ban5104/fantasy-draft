#!/usr/bin/env python3
"""
Golden Master Test for Draft Optimizer

This test runs the optimizer with a fixed seed and saves the CSV outputs
to a tests/golden/ directory, then provides comparison functions for regression testing.
"""

import os
import sys
import pandas as pd
import shutil
import numpy as np
from pathlib import Path

# Add scripts directory to Python path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import optimizer functions directly
try:
    from dp_draft_optimizer_debug import (
        load_and_merge_data, monte_carlo_survival_realistic, 
        get_position_survival_matrix, export_mc_results_to_csv,
        dp_optimize, SNAKE_PICKS, pos_sorted, clear_capture_examples, Player
    )
    import logging
except ImportError as e:
    print(f"Error importing optimizer functions: {e}")
    print("Make sure you're running from the scripts directory or project root")
    sys.exit(1)


def create_golden_master():
    """Generate golden master outputs by running optimizer directly with fixed seed."""
    print("Generating golden master outputs...")
    
    # Ensure tests/golden directory exists
    golden_dir = project_root / "tests" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Change to project root directory for proper file access
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        
        # Set fixed seed for reproducibility
        np.random.seed(42)
        print(f"Random seed set to: 42")
        
        # Clear previous examples
        clear_capture_examples()
        
        logger.info("Loading player data...")
        
        # Load players using simplified data loading
        players = load_and_merge_data()
        print(f"Loaded {len(players)} players")
            
        # Run Monte Carlo simulation with stable parameters (5000 sims)
        sims = 5000
        data_source = "espn"
        
        logger.info(f"Running {sims} Monte Carlo simulations...")
        player_survival = monte_carlo_survival_realistic(
            players, sims, export_simulation_data=True, data_source=data_source, enhanced_stats=False
        )
        
        # Export CSV results
        export_mc_results_to_csv(
            players, player_survival, SNAKE_PICKS, sims, data_source, 42, enhanced_stats=False
        )
        
        # Move generated CSV files to tests/golden directory
        source_dir = project_root / "data" / "output-simulations"
        data_files = [
            f"mc_player_survivals_{data_source}.csv",
            f"mc_position_summary_{data_source}.csv", 
            f"mc_config_{data_source}.csv"
        ]
        
        moved_files = 0
        for filename in data_files:
            source_path = source_dir / filename
            if source_path.exists():
                golden_path = golden_dir / filename
                shutil.copy2(source_path, golden_path)
                moved_files += 1
                print(f"Saved golden master: {golden_path}")
            else:
                print(f"WARNING: Expected output file not found: {source_path}")
        
        if moved_files > 0:
            print(f"✓ Golden master created with {moved_files} files")
            return True
        else:
            print("✗ No output files found - golden master creation failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Golden master generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def compare_with_golden_master(tolerance=0.001):
    """Compare current CSV outputs with golden master files."""
    print("Comparing current outputs with golden master...")
    
    golden_dir = project_root / "tests" / "golden"
    if not golden_dir.exists():
        print("ERROR: Golden master directory not found. Run create_golden_master() first.")
        return False
    
    try:
        # Change to project root directory for proper file access
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        
        # Set same fixed seed
        np.random.seed(42)
        
        # Clear previous examples
        clear_capture_examples()
        
        logger.info("Loading player data...")
        
        # Load players using same method as golden master
        players = load_and_merge_data()
            
        # Run Monte Carlo simulation with same parameters
        sims = 5000
        data_source = "espn"
        
        logger.info(f"Running {sims} Monte Carlo simulations...")
        player_survival = monte_carlo_survival_realistic(
            players, sims, export_simulation_data=True, data_source=data_source, enhanced_stats=False
        )
        
        # Export CSV results
        export_mc_results_to_csv(
            players, player_survival, SNAKE_PICKS, sims, data_source, 42, enhanced_stats=False
        )
        
    except Exception as e:
        print(f"ERROR: Test run failed: {e}")
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    # Compare each CSV file
    comparison_results = []
    csv_files = ["mc_player_survivals_espn.csv", "mc_position_summary_espn.csv", "mc_config_espn.csv"]
    
    for filename in csv_files:
        golden_path = golden_dir / filename
        current_path = project_root / "data" / "output-simulations" / filename
        
        if not golden_path.exists():
            print(f"WARNING: Golden master file missing: {golden_path}")
            continue
            
        if not current_path.exists():
            print(f"ERROR: Current output file missing: {current_path}")
            comparison_results.append(False)
            continue
        
        # Load and compare DataFrames
        try:
            golden_df = pd.read_csv(golden_path)
            current_df = pd.read_csv(current_path)
            
            # Check shape
            if golden_df.shape != current_df.shape:
                print(f"✗ {filename}: Shape mismatch - Golden: {golden_df.shape}, Current: {current_df.shape}")
                comparison_results.append(False)
                continue
            
            # Check numeric columns within tolerance
            numeric_cols = golden_df.select_dtypes(include=['number']).columns
            differences_found = False
            
            for col in numeric_cols:
                if col in current_df.columns:
                    max_diff = abs(golden_df[col] - current_df[col]).max()
                    if max_diff > tolerance:
                        print(f"✗ {filename}: Column '{col}' max difference {max_diff:.6f} exceeds tolerance {tolerance}")
                        differences_found = True
            
            # Check non-numeric columns for exact match
            non_numeric_cols = golden_df.select_dtypes(exclude=['number']).columns
            for col in non_numeric_cols:
                if col in current_df.columns:
                    if not golden_df[col].equals(current_df[col]):
                        print(f"✗ {filename}: Column '{col}' has non-matching values")
                        differences_found = True
            
            if not differences_found:
                print(f"✓ {filename}: Matches golden master")
                comparison_results.append(True)
            else:
                comparison_results.append(False)
                
        except Exception as e:
            print(f"ERROR comparing {filename}: {e}")
            comparison_results.append(False)
    
    # Summary
    if all(comparison_results):
        print("✓ All files match golden master - No regressions detected!")
        return True
    else:
        print("✗ Some files differ from golden master - Potential regression!")
        return False


def main():
    """Main function to handle command line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/test_golden_master.py create    # Generate golden master")
        print("  python scripts/test_golden_master.py compare   # Compare with golden master")
        return
    
    action = sys.argv[1].lower()
    
    if action == "create":
        success = create_golden_master()
        sys.exit(0 if success else 1)
    elif action == "compare":
        success = compare_with_golden_master()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown action: {action}")
        print("Use 'create' or 'compare'")
        sys.exit(1)


if __name__ == "__main__":
    main()