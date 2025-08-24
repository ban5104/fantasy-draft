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
# Quick results (100 simulations, basic output)
python scripts/dp_draft_optimizer_debug.py --quick

# Standard mode (5000 simulations, full analytics and exports) - DEFAULT
python scripts/dp_draft_optimizer_debug.py --standard

# Debug mode (standard + debug output and visualizations)
python scripts/dp_draft_optimizer_debug.py --debug

# Specify your draft position (1-14, auto-calculates snake picks)
python scripts/dp_draft_optimizer_debug.py --standard --position 8

# With envelope projections for uncertainty analysis
python scripts/dp_draft_optimizer_debug.py --standard --envelope-file data/my_projections.csv

# Reproducible results with seed
python scripts/dp_draft_optimizer_debug.py --standard --position 5 --seed 42
```

### Draft Cheat Sheet
```bash
# Update cheat sheet with latest data from all sources
python scripts/update_cheat_sheet.py

# Verbose output to see data merge process
python scripts/update_cheat_sheet.py -v
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Key Files

- `scripts/dp_draft_optimizer_debug.py` - Main optimization engine (self-contained)
- `scripts/update_cheat_sheet.py` - Draft cheat sheet generator 
- `data/probability-models-draft/` - ESPN projections, ADP data, draft results (4 data sources)
- `data/draft_day_cheat_sheet.csv` - Generated cheat sheet with merged data from all sources
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
- `DRAFT_POSITION` - Your pick number in round 1 (1-14, set via `--position` flag)
- `SNAKE_PICKS` - Auto-calculated from draft position (e.g., pos 5: [5, 24, 33, 52, 61, 80, 89])
- `POSITION_LIMITS` - Roster targets ({'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1})
- `RANDOMNESS_LEVEL` - Draft unpredictability (0.1-0.7, default 0.3)
- `CANDIDATE_POOL_SIZE` - Players considered per pick (5-25, default 15)

### Envelope Integration Parameters
- `ENVELOPE_FILE` - Path to envelope projections CSV file (default: None)
- `USE_ENVELOPES` - Enable envelope functionality and analytics capture (default: True)
- `EXPORT_DIR` - Directory for analytics exports (default: "data/output-simulations")
- `EXPORT_FORMAT` - Export format for analytics data (default: "parquet", fallback to CSV)

## Data Format

- ESPN projections: CSV with player_name, position, overall_rank
  - Available: `espn_projections_20250814.csv`, `espn_algorithm_20250824.csv`
  - Configurable via `--espn-file` parameter (optional)
- Fantasy rankings: `rankings_top300_20250814.csv` with PLAYER, FANTASY_PTS
- Draft cheat sheet sources (4 files in `data/probability-models-draft/`):
  - `espn_algorithm_*.csv` → ESPN_ALG column
  - `espn_projections_*.csv` → ESPN_PROJ column  
  - `realtime_adp_*.csv` → ADP_SLEEPER column (Sleeper ADP)
  - `actual_draft_results_*.csv` → ACTUAL_DRAFT_YYYYMMDD column
  - All use standardized format: `overall_rank,position,position_rank,player_name,team`
  - Script automatically finds latest files by timestamp (91.2% data coverage)
- Envelope projections: CSV with flexible column names for projection ranges
  - Player: `name`, `player`, `player_name`
  - Position: `pos`, `position` 
  - Low bound: `low`, `floor`, `p10`
  - Projection: `proj`, `projection`, `mode`, `median`, `p50`, `center`
  - High bound: `high`, `ceiling`, `p90`
  - Enables uncertainty analysis with derived metrics (safety index, volatility index)
- All data follows standardized format with 99.6% exact matching

### Analytics Output Files (standard/debug modes)
- `pick_candidates.csv` - Top candidates per pick with envelope metrics
- `value_decay.csv` - Value dropoff analysis between consecutive picks
- `pos_outlook.csv` - Positional availability trends across rounds
- `run_metadata.json` - Execution metadata and file hashes for reproducibility

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