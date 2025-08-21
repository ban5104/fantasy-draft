# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fantasy football draft optimization tool that uses dynamic programming and Monte Carlo simulation to determine optimal draft strategies. The system combines survival probability modeling with position-based draft value calculations to recommend which position to draft at each pick in a snake draft format.

## Core Architecture

The codebase implements a **Dynamic Programming over Positions** approach rather than enumerating individual players:

- **State Space**: 48 total states defined by (pick_index, rb_count, wr_count, qb_count, te_count)
- **Transition Function**: `F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}`
- **Objective**: Maximize expected fantasy points across all draft picks

### Key Components

1. **Data Pipeline** (`load_and_merge_data`): Merges ESPN projections with fantasy point rankings using fuzzy string matching
2. **Monte Carlo Simulation** (`monte_carlo_survival_realistic`): Models draft randomness to compute player survival probabilities 
3. **Ladder Expected Value** (`ladder_ev_debug`): Calculates expected points for drafting a position at a specific pick/slot
4. **DP Solver** (`dp_optimize`): Backward induction optimization with memoization

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

### Running the Optimizer

```bash
# Basic optimization with default settings
python scripts/dp_draft_optimizer_debug.py

# With custom simulation parameters
python scripts/dp_draft_optimizer_debug.py --sims 5000 --randomness 0.3 --pool-size 15

# Export results for analysis
python scripts/dp_draft_optimizer_debug.py --export-csv --export-simulations

# Generate visualizations (requires matplotlib)
python scripts/dp_draft_optimizer_debug.py --visualize --save-plots
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

Debug mode provides detailed pick-by-pick analysis:
- Expected value calculations for each position at each pick
- Player survival probabilities 
- Optimal decision reasoning
- Delta analysis (value difference between current pick vs. waiting)

## Monte Carlo Simulation

The simulation models realistic draft behavior:
- Teams select from top N candidates with weighted randomness
- Pure best-player-available strategy (no position scarcity modeling)
- Configurable randomness and candidate pool size
- Exports individual pick data for scatter plot analysis

## Output Files

When using `--export-csv` flag:
- `mc_player_survivals.csv` - Player-level survival probabilities
- `mc_position_summary.csv` - Position-level statistics
- `mc_simulation_picks.csv` - Individual simulation pick data
- `mc_config.csv` - Simulation metadata

## Validation

The system includes several validation checks:
- Fuzzy matching quality warnings for player name mismatches
- Survival probability matrix validation
- DP state boundary checking
- Data file existence verification