# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fantasy football draft optimization tool that uses dynamic programming and Monte Carlo simulation to determine optimal draft strategies. The system combines survival probability modeling with position-based draft value calculations to recommend which position to draft at each pick in a snake draft format.

**Recent Major Enhancements**: The optimizer has undergone significant correctness fixes and performance improvements, including proper fantasy point ordering, 5-10x Monte Carlo speedup, enhanced debugging capabilities, and full reproducibility with seed support.

## Core Architecture

The codebase implements a **Dynamic Programming over Positions** approach rather than enumerating individual players:

- **State Space**: 48 total states defined by (pick_index, rb_count, wr_count, qb_count, te_count)
- **Transition Function**: `F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}`
- **Objective**: Maximize expected fantasy points across all draft picks

### Key Components

1. **Data Pipeline** (`load_and_merge_data`): Merges ESPN projections with fantasy point rankings using fuzzy string matching, with D/ST and K filtering
2. **Monte Carlo Simulation** (`monte_carlo_survival_realistic`): Rank-weighted Gaussian noise selection with 5-10x performance optimization 
3. **Ladder Expected Value** (`ladder_ev_debug`): Calculates expected points with correct fantasy point ordering and matrix alignment
4. **DP Solver** (`dp_optimize`): Backward induction optimization with early termination and enhanced memoization

## File Structure

- `archived/dp_draft_optimizer.py` - Original simplified implementation (~200 lines)
- `scripts/dp_draft_optimizer_debug.py` - Full-featured version with debug output and visualization
- `data/` - Player data and simulation outputs
- `jupyter-notebooks/` - Analysis notebooks and visualization tools
- `specs/` - Theoretical foundation and planning documents (**READ FIRST**)

## Theoretical Foundation (specs/ directory)

**Start here for context.** The `specs/` directory contains the mathematical foundation and high-level strategy documents that explain the "why" behind this implementation:

- `plan.md` - MVP framework and implementation roadmap
- `therory.md` - Mathematical formulation and theoretical background

**Key concepts defined in specs/:**
- Dynamic Programming over Positions approach
- Monte Carlo survival probability modeling
- Ladder expected value calculations
- State space design decisions

These documents explain the theoretical underpinnings that drive the code architecture. Understanding the mathematical model in `specs/` is essential for:
- Making informed changes to the optimization algorithm
- Understanding why certain implementation choices were made
- Extending the system with new features or constraints
- Debugging unexpected optimization results

**Developer workflow recommendation:**
1. Read `specs/plan.md` for the high-level approach
2. Review `specs/therory.md` for mathematical details
3. Examine `scripts/dp_draft_optimizer_debug.py` for implementation
4. Use `jupyter-notebooks/` for analysis and validation

## Development Commands

### Standard Test

```bash
# Recommended standard test - comprehensive optimization with quality checks
python scripts/test_optimizer.py
```

This runs: `python3 scripts/dp_draft_optimizer_debug.py --sims 10000 --export-csv`

### Running the Optimizer

**Mode Presets (Recommended):**

```bash
# Fast mode - quick results (100 simulations)
python scripts/dp_draft_optimizer_debug.py --mode fast

# Stable mode - production quality (5000 simulations, CSV exports)
python scripts/dp_draft_optimizer_debug.py --mode stable

# Debug mode - full analysis with visualizations (1000 simulations)
python scripts/dp_draft_optimizer_debug.py --mode debug
```

**Advanced Usage:**

```bash
# Reproducible results with seed
python scripts/dp_draft_optimizer_debug.py --mode stable --seed 42

# Parameter robustness testing
python scripts/dp_draft_optimizer_debug.py --stability-sweep

# Custom simulation parameters (overrides mode presets)
python scripts/dp_draft_optimizer_debug.py --sims 5000 --randomness 0.3 --pool-size 15

# Full feature set
python scripts/dp_draft_optimizer_debug.py --export-csv --export-simulations --visualize --save-plots
```

### Jupyter Analysis

```bash
# Start Jupyter for Monte Carlo analysis
jupyter notebook jupyter-notebooks/monte_carlo_statistical_analysis.ipynb

# View visualization dashboard
jupyter notebook jupyter-notebooks/monte_carlo_visualization.ipynb
```

### Dependencies

Install with: `pip install -r requirements.txt`

Core dependencies: pandas, numpy, scipy, rapidfuzz, matplotlib, seaborn, plotly, notebook

## Configuration

Key parameters in `scripts/dp_draft_optimizer_debug.py`:

- `SNAKE_PICKS`: Your actual draft pick positions (currently set for 14-team league: [5, 24, 33, 52, 61, 80, 89])
- `POSITION_LIMITS`: Roster construction targets ({'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1})
- `RANDOMNESS_LEVEL`: Draft unpredictability (0.1-0.7, default 0.3)
- `CANDIDATE_POOL_SIZE`: How many top players teams consider per pick (5-25, default 15)

## Data Requirements

The system expects two CSV files in `data/`:
- `espn_projections_20250814.csv` - ESPN player projections with columns: player_name, position, overall_rank
- `rankings_top300_20250814.csv` - Fantasy point projections with columns: PLAYER, FANTASY_PTS

## Debug Mode

Debug mode provides comprehensive pick-by-pick analysis:
- Expected value calculations for each position at each pick
- Player survival probabilities with availability display (e.g., "P=0.82 best-available")
- Optimal decision reasoning with counterfactual analysis
- Delta analysis (immediate value difference) AND total DP value for complete transparency
- Stability sweep for parameter robustness testing

## Monte Carlo Simulation

The simulation models realistic draft behavior with significant performance improvements:
- Rank-weighted Gaussian noise selection: `(1.0 / (rank + 1)) * max(0.1, normal(1.0, RANDOMNESS_LEVEL))`
- RANDOMNESS_LEVEL controls the standard deviation of the normal distribution for draft unpredictability
- 5-10x speedup from boolean masking optimization (fixed O(n) bottleneck)
- Fully reproducible results with seed parameter support
- Monotonic survival probability validation with automatic smoothing
- Configurable randomness and candidate pool size
- Exports individual pick data for scatter plot analysis

## Output Files

When using `--export-csv` flag (automatically enabled in stable/debug modes):
- `mc_player_survivals.csv` - Player-level survival probabilities
- `mc_position_summary.csv` - Position-level statistics
- `mc_simulation_picks.csv` - Individual simulation pick data
- `mc_config.csv` - Simulation metadata including seed for reproducibility

## Validation

The system includes comprehensive validation and correctness checks:
- **Core Logic Validation**: Players correctly sorted by fantasy points (not ESPN order)
- **Matrix Alignment**: Survival probability matrices perfectly aligned with ladder calculations  
- **Data Quality**: D/ST and K positions filtered out to reduce fuzzy matching noise
- **Monotonic Survival**: Survival probabilities validated and smoothed to ensure mathematical correctness
- **Fuzzy Matching**: Quality warnings for player name mismatches with detailed mismatch reporting
- **DP State Boundary**: Early termination when roster positions are filled
- **Reproducibility**: Seed validation ensures identical results across runs