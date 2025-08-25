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

# With envelope projections for uncertainty analysis
python scripts/dp_draft_optimizer_debug.py --mode stable --envelope-file data/my_projections.csv

# Enhanced analytics data capture
python scripts/dp_draft_optimizer_debug.py --mode stable --capture-analytics

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
- `DRAFT_POSITION` - Your position in draft (1-14, default 5)
- `LEAGUE_SIZE` - Number of teams (default 14)
- `POSITION_LIMITS` - Roster targets ({'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1})
- `RANDOMNESS_LEVEL` - Draft unpredictability (0.1-0.7, default 0.3)
- `CANDIDATE_POOL_SIZE` - Players considered per pick (default 25)

### Enhanced Decision Support Parameters
- `EPSILON_THRESHOLD` - Show strategies within this % of optimal (default 0.033 = 3.3%)
- `K_BEST_DEPTH` - Alternative paths to explore per pick (default 10)

### Analytics and Export Parameters
- `USE_ANALYTICS` - Enable detailed analytics capture (default: True)
- `EXPORT_DIR` - Directory for analytics exports (default: "data/output-simulations")
- `EXPORT_FORMAT` - Export format for analytics data (default: "parquet", fallback to CSV)
- `ENVELOPE_FILE` - Path to envelope projections CSV file (default: None)

## Data Format

- ESPN projections: CSV with player_name, position, overall_rank
  - Available: `espn_projections_20250814.csv`, `espn_algorithm_20250824.csv`
  - Configurable via `--espn-file` parameter
- Fantasy rankings: `rankings_top300_20250814.csv` with PLAYER, FANTASY_PTS
- Envelope projections: CSV with flexible column names for projection ranges
  - Player: `name`, `player`, `player_name`
  - Position: `pos`, `position` 
  - Low bound: `low`, `floor`, `p10`
  - Projection: `proj`, `projection`, `mode`, `median`, `p50`, `center`
  - High bound: `high`, `ceiling`, `p90`
  - Enables uncertainty analysis with derived metrics (safety index, volatility index)
- Additional data: ADP, actual draft results in `probability-models-draft/`
- All data follows standardized format with 99.6% exact matching

### Analytics Output Files (when analytics enabled)
- `pick_candidates.csv` - Top candidates per pick with availability probabilities and envelope metrics (if available)
- `value_decay.csv` - Value dropoff analysis between consecutive picks by position
- `pos_outlook.csv` - Positional availability trends across draft rounds
- `run_metadata.json` - Execution metadata and file hashes for reproducibility
- Standard Monte Carlo exports: `mc_config_*.csv`, `mc_player_survivals_*.csv`, `mc_position_summary_*.csv`

## Development Workflow

1. Read `specs/` directory first for mathematical context
2. Run regression tests before changes: `python scripts/tests/test_golden_master.py`
3. Make changes to main script only (keep self-contained)
4. Test changes: `python scripts/tests/test_optimizer.py`
5. Run unit tests: `python scripts/tests/test_favorites.py` 
6. Use notebooks only for visualization and analysis of CSV outputs

## Enhanced Features

### Decision Support Enhancements
- **ε-Optimal Plans Menu**: Multiple draft strategies within 3.3% of optimal EV (configurable via EPSILON_THRESHOLD)
- **Pick Regret Analysis**: Compares alternative position choices with regret percentages  
- **Flexibility Index**: Entropy-based scoring (0.0-1.0) showing decision flexibility
- **Time-to-Cliff Analysis**: Warns when positions face significant value drops
- **Contingency Playbooks**: Primary/secondary/tertiary recommendations per pick
- **Risk-Adjusted Variants**: Floor-focused and upside-focused strategies (requires envelope data)
- **K-Best Path Exploration**: Configurable depth for alternative strategy discovery

### Enhanced Output Sections
1. **Draft Plan Menu**: Shows multiple viable strategies (e.g., "Plan A: RB-QB-RB-WR... EV 1380.9")
2. **Per-Pick Analysis**: Regret tables, flexibility scores, cliff warnings, contingency trees
3. **Risk Variants**: Conservative vs aggressive approaches when envelope data available

### Sample Enhanced Output
```
=== DRAFT PLAN MENU (ε=3.3%) ===
Plan A: RB-QB-RB-WR-RB-WR-TE    EV 1380.9  (baseline)
Plan B: RB-WR-RB-WR-RB-QB-TE    EV 1374.1  (-0.5%, safer)
Plan C: WR-RB-QB-WR-RB-WR-TE    EV 1370.2  (-0.8%, WR heavy)

PICK 5 ANALYSIS
Flexibility Index: 0.71 (many viable options)
Windows: RB cliff in 1 pick, WR safe for 2 picks

REGRET TABLE:
Position    EV     Regret    Notes
RB        380.9    0.0%     (optimal)
WR        375.2   -1.5%     Strong alternative
QB        365.1   -4.2%     Early but viable

CONTINGENCY PLAYBOOK:
🎯 PRIMARY: RB → Saquon Barkley, Jonathan Taylor
📋 IF GONE: WR → CeeDee Lamb, Tyreek Hill (-1.5% EV)
🔄 LAST RESORT: QB → Josh Allen, Lamar Jackson (-4.2% EV)
```

## Current Project State

- **Enhanced decision support**: Comprehensive draft-day analysis with multiple strategies and contingency planning
- **Cleaned architecture**: Removed old simulation outputs and unused files
- **Consolidated data**: All probability models in `data/probability-models-draft/`
- **Simplified structure**: 3 focused Jupyter notebooks for analysis
- **Improved testing**: Golden master tests in `tests/golden/` with unit tests in `scripts/tests/`