#!/usr/bin/env python3
"""
Standard test script for the fantasy draft optimizer.
Runs comprehensive optimization with quality checks and data export.
Enhanced with unit tests for new enhancement features.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add the scripts directory to Python path to import the optimizer functions
sys.path.append(str(Path(__file__).parent.parent))

# Import the new enhancement functions for testing
from dp_draft_optimizer_debug import (
    get_epsilon_optimal_plans,
    compute_pick_regret,
    compute_cliff_windows,
    compute_flexibility_index,
    build_contingency_tree,
    show_contingency_tree,
    get_top_available_players
)


def run_optimizer_test():
    """Run the standard optimizer test with consistent parameters."""

    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # Standard test command with consistent parameters
    cmd = [
        "python3",
        "scripts/dp_draft_optimizer_debug.py",
        "--sims",
        "10000",
        "--export-csv",
        "--espn-file",
        "data/probability-models-draft/espn_projections_20250814.csv",
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


def test_enhanced_features():
    """Test new enhancement features (Phase 1 & 2 validations)."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCEMENT FEATURES")
    print("=" * 60)
    
    try:
        # Initialize minimal test data for unit tests
        test_counts = {"RB": 0, "WR": 0, "QB": 0, "TE": 0}
        
        # Test 1: ε-optimal plans
        print("Testing get_epsilon_optimal_plans()...")
        plans = get_epsilon_optimal_plans(epsilon=0.02)
        assert len(plans) >= 1, "Should return at least one plan"
        if len(plans) > 1:
            assert plans[0]['total_value'] >= plans[1]['total_value'], "Plans should be sorted by EV"
        print("✅ ε-optimal plans working correctly")
        
        # Test 2: Regret analysis
        print("Testing compute_pick_regret()...")
        regret = compute_pick_regret(0, test_counts)
        assert len(regret) >= 2, "Should return regret for multiple positions"
        assert regret[0]['regret'] <= regret[1]['regret'], "Should be sorted by regret (ascending)"
        assert all('position' in r and 'regret_pct' in r for r in regret), "Should have required fields"
        print("✅ Pick regret analysis working correctly")
        
        # Test 3: Cliff windows
        print("Testing compute_cliff_windows()...")
        cliff_windows = compute_cliff_windows(0, test_counts)
        assert isinstance(cliff_windows, dict), "Should return dictionary"
        for pos, window_info in cliff_windows.items():
            assert pos in ["RB", "WR", "QB", "TE"], "Should only contain valid positions"
            assert 'picks_to_cliff' in window_info, "Should have picks_to_cliff field"
            assert 'safe_window' in window_info, "Should have safe_window field"
            assert 'cliff_drop_pct' in window_info, "Should have cliff_drop_pct field"
        print("✅ Cliff windows analysis working correctly")
        
        # Test 4: Flexibility calculation
        print("Testing compute_flexibility_index()...")
        test_position_values = {
            "RB": {"total_value": 100}, 
            "WR": {"total_value": 95}, 
            "QB": {"total_value": 85}
        }
        flex = compute_flexibility_index(test_position_values)
        assert 0 <= flex <= 1, f"Flexibility should be between 0 and 1, got {flex}"
        
        # Test edge cases
        flex_equal = compute_flexibility_index({"RB": {"total_value": 100}, "WR": {"total_value": 100}})
        assert flex_equal == 1.0, "Equal values should give maximum flexibility"
        
        flex_single = compute_flexibility_index({"RB": {"total_value": 100}})
        assert flex_single == 0.0, "Single option should give zero flexibility"
        
        flex_empty = compute_flexibility_index({})
        assert flex_empty == 0.0, "Empty dict should give zero flexibility"
        print("✅ Flexibility index working correctly")
        
        # Test 5: Contingency trees
        print("Testing build_contingency_tree()...")
        tree = build_contingency_tree(0, test_counts)
        assert isinstance(tree, dict), "Should return dictionary"
        assert 'primary' in tree, "Should have primary option"
        assert 'secondary' in tree, "Should have secondary option"
        assert 'tertiary' in tree, "Should have tertiary option"
        
        # Validate tree structure
        for key in ['primary', 'secondary', 'tertiary']:
            if tree[key].get('position'):
                assert 'players' in tree[key], f"{key} should have players list"
                assert isinstance(tree[key]['players'], list), f"{key} players should be list"
        print("✅ Contingency trees working correctly")
        
        # Test 6: Top available players
        print("Testing get_top_available_players()...")
        players = get_top_available_players("RB", 5, n=3)
        assert isinstance(players, list), "Should return list"
        assert len(players) <= 3, "Should not exceed requested count"
        print("✅ Top available players working correctly")
        
        print("\n" + "=" * 60)
        print("✅ All enhanced features working correctly!")
        print("Phase 1 Features: ✅ ε-optimal plans, ✅ regret analysis, ✅ cliff windows")
        print("Phase 2 Features: ✅ flexibility index, ✅ contingency trees")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Enhancement features test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run both integration and unit tests
    integration_success = run_optimizer_test()
    enhancement_success = test_enhanced_features()
    
    overall_success = integration_success and enhancement_success
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    sys.exit(0 if overall_success else 1)
