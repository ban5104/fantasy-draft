# Fantasy Football Draft Optimizer - Claude Instructions

## Project Context

This is a fantasy football draft optimization tool using dynamic programming and Monte Carlo simulation. The entire system is contained in a single self-contained script: `scripts/dp_draft_optimizer_debug.py`.

## Architecture Constraints

- **Keep main script unified** - Do not break `scripts/dp_draft_optimizer_debug.py` into multiple files
- **Notebooks for visualization only** - No complex calculations in Jupyter notebooks
- **Single source of truth** - All optimization logic stays in the main script
- **No modular refactoring** - The unified architecture is intentional

## Development Commands

### Standard Testing
```bash
# Run golden master regression tests before any changes
python scripts/tests/test_golden_master.py

# Standard comprehensive test
python scripts/tests/test_optimizer.py

# Unit tests for favorites functionality  
python scripts/tests/test_favorites.py
```

### Running the Optimizer
```bash
# Quick results (100 simulations)
python scripts/dp_draft_optimizer_debug.py --mode fast

# Production quality (5000 simulations, CSV exports)
python scripts/dp_draft_optimizer_debug.py --mode stable

# Full analysis with visualizations
python scripts/dp_draft_optimizer_debug.py --mode debug

# Reproducible results
python scripts/dp_draft_optimizer_debug.py --mode stable --seed 42
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Key Files

- `scripts/dp_draft_optimizer_debug.py` - Main optimization engine (self-contained)
- `data/probability-models-draft/` - ESPN projections, ADP data, draft results
- `data/rankings_top300_20250814.csv` - Fantasy point rankings
- `specs/plan.md` - Mathematical approach and theory
- `specs/therory.md` - Detailed mathematical formulation
- `tests/golden/` - Golden master regression test outputs
- `jupyter-notebooks/` - Analysis and visualization (3 notebooks)
  - `monte_carlo_statistical_analysis.ipynb` - Statistical analysis
  - `model_comparison_analysis.ipynb` - Strategy comparisons  
  - `value_dropoff_charts.ipynb` - Visualization charts

## Configuration Parameters

Key variables in main script:
- `SNAKE_PICKS` - Draft pick positions (currently 14-team: [5, 24, 33, 52, 61, 80, 89])
- `POSITION_LIMITS` - Roster targets ({'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1})
- `RANDOMNESS_LEVEL` - Draft unpredictability (0.1-0.7, default 0.3)
- `CANDIDATE_POOL_SIZE` - Players considered per pick (5-25, default 15)

## Data Format

- ESPN projections: CSV with player_name, position, overall_rank
  - Available: `espn_projections_20250814.csv`, `espn_algorithm_20250824.csv`
  - Configurable via `--espn-file` parameter
- Fantasy rankings: `rankings_top300_20250814.csv` with PLAYER, FANTASY_PTS
- Additional data: ADP, actual draft results in `probability-models-draft/`
- All data follows standardized format with 99.6% exact matching

## Development Workflow

1. Read `specs/` directory first for mathematical context
2. Run regression tests before changes: `python scripts/tests/test_golden_master.py`
3. Make changes to main script only (keep self-contained)
4. Test changes: `python scripts/tests/test_optimizer.py`
5. Run unit tests: `python scripts/tests/test_favorites.py` 
6. Use notebooks only for visualization and analysis of CSV outputs

## Current Project State

- **Cleaned architecture**: Removed old simulation outputs and unused files
- **Consolidated data**: All probability models in `data/probability-models-draft/`
- **Simplified structure**: 3 focused Jupyter notebooks for analysis
- **Improved testing**: Golden master tests in `tests/golden/` with unit tests in `scripts/tests/`