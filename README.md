# Fantasy Football Draft Optimizer

A data-driven draft optimization tool that uses dynamic programming and Monte Carlo simulation to determine optimal fantasy football draft strategies. Instead of evaluating individual players, this system uses a novel "Dynamic Programming over Positions" approach to maximize expected fantasy points across all draft picks.

**Recent Major Enhancements**: Architectural refactoring with simplified data loading, golden master testing framework, elimination of circular imports, and significant correctness fixes with 5-10x performance improvements.

## Why This Approach

Traditional draft tools focus on player rankings and ADP (Average Draft Position). This optimizer goes deeper by:

- **Modeling draft randomness** with Monte Carlo simulation to predict player availability
- **Optimizing position selection** rather than specific players at each pick
- **Using dynamic programming** to find globally optimal strategies, not greedy pick-by-pick decisions
- **Accounting for roster construction** with position limits and snake draft dynamics

The result is a strategy that tells you which **position** to target at each of your picks, maximizing your team's expected point total.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fast optimization (100 simulations, quick results)
python scripts/dp_draft_optimizer_debug.py --mode fast

# Production quality (5000 simulations, CSV exports)
python scripts/dp_draft_optimizer_debug.py --mode stable

# Full analysis with visualizations (1000 simulations)
python scripts/dp_draft_optimizer_debug.py --mode debug
```

## Installation

**Prerequisites:** Python 3.8+

```bash
git clone <repository-url>
cd fantasy-draft
pip install -r requirements.txt
```

**Core Dependencies:**
- pandas, numpy, scipy - Data processing and optimization
- rapidfuzz - Player name matching between datasets
- matplotlib, seaborn, plotly - Visualization and analysis
- notebook - Jupyter analysis environment

## Usage

### Mode Presets (Recommended)

```bash
# Fast mode - quick results for testing (100 simulations)
python scripts/dp_draft_optimizer_debug.py --mode fast

# Stable mode - production quality with exports (5000 simulations)
python scripts/dp_draft_optimizer_debug.py --mode stable

# Debug mode - full analysis with visualizations (1000 simulations)
python scripts/dp_draft_optimizer_debug.py --mode debug
```

### Advanced Usage

```bash
# Reproducible results with seed parameter
python scripts/dp_draft_optimizer_debug.py --mode stable --seed 42

# Parameter robustness testing
python scripts/dp_draft_optimizer_debug.py --stability-sweep

# Custom parameters (overrides mode presets)
python scripts/dp_draft_optimizer_debug.py --sims 10000 --randomness 0.4 --pool-size 20

# Custom ESPN data source
python scripts/dp_draft_optimizer_debug.py --mode stable --espn-file data/espn_algorithm_20250824.csv

# Full feature set
python scripts/dp_draft_optimizer_debug.py --export-csv --export-simulations --visualize --save-plots
```

### Data Analysis

```bash
# Launch Jupyter for Monte Carlo statistical analysis
jupyter notebook jupyter-notebooks/monte_carlo_statistical_analysis.ipynb

# Model comparison and strategy analysis
jupyter notebook jupyter-notebooks/model_comparison_analysis.ipynb

# Value dropoff visualization
jupyter notebook jupyter-notebooks/value_dropoff_charts.ipynb
```

## How It Works

### 1. Data Pipeline
Uses simplified hierarchical matching system with exact match priority, achieving 99.6% exact matches. Automatically filters D/ST and K positions with configurable ESPN data sources.

### 2. Monte Carlo Simulation
Simulates thousands of draft scenarios with rank-weighted Gaussian noise selection and 5-10x performance optimization to compute **survival probabilities** - the likelihood each player will be available at each of your picks.

### 3. Dynamic Programming Optimization
Uses backward induction to find the optimal position to draft at each pick:

- **State Space:** 48 states defined by `(pick_index, rb_count, wr_count, qb_count, te_count)`
- **Objective:** Maximize total expected fantasy points across all picks
- **Constraints:** Position limits (e.g., max 3 RB, 2 WR, 1 QB, 1 TE)

### 4. Position Ladder Expected Value
Calculates the expected points from drafting the "next best available" player at each position with correct fantasy point ordering and survival probability alignment.

## Configuration

Edit key parameters in `scripts/dp_draft_optimizer_debug.py`:

```python
# Your draft pick positions (snake draft)
SNAKE_PICKS = [5, 24, 33, 52, 61, 80, 89]  # 14-team league example

# Roster construction targets
POSITION_LIMITS = {'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1}

# Simulation parameters
RANDOMNESS_LEVEL = 0.3      # Draft unpredictability (0.1-0.7)
CANDIDATE_POOL_SIZE = 15    # Players each team considers per pick
```

## Data Requirements

The system uses flexible data loading with hierarchical matching:

- **ESPN Projections**: Any CSV file with columns: `player_name`, `position`, `overall_rank`
  - Configurable via `--espn-file` parameter  
  - Available data: `data/probability-models-draft/espn_projections_20250814.csv`, `espn_algorithm_20250824.csv`
- **Fantasy Rankings**: `data/rankings_top300_20250814.csv` with columns: `PLAYER`, `FANTASY_PTS`
- **Additional Data**: ADP data, actual draft results, and other probability models in `data/probability-models-draft/`

**Data Quality**: 99.6% exact matches with 4-tier hierarchical matching system and 92% fuzzy threshold.

## File Structure

```
fantasy-draft-fav-players/
├── scripts/
│   ├── dp_draft_optimizer_debug.py    # Main optimizer (self-contained)
│   └── tests/                         # Unit and integration tests
├── data/
│   ├── probability-models-draft/      # ESPN projections, ADP, draft results
│   ├── rankings_top300_20250814.csv   # Fantasy point rankings
│   └── draft_day_cheat_sheet.txt      # Quick reference
├── jupyter-notebooks/                 # Analysis and visualization (3 notebooks)
├── tests/golden/                      # Golden master regression tests
├── specs/                             # Mathematical theory and planning
├── archived/                          # Previous implementations
└── requirements.txt                   # Python dependencies
```

## Developer Guide

For implementation details, mathematical background, and development workflows, see [CLAUDE.md](CLAUDE.md).

For theoretical foundation and high-level strategy planning, see the `specs/` directory.

## Output

The optimizer produces:

1. **Optimal Strategy:** Which position to draft at each of your picks
2. **Enhanced Debug Analysis:** Both immediate decision value (Delta) and total DP value with counterfactual analysis
3. **Player Availability:** Survival probabilities with clear availability displays (e.g., "P=0.82 best-available")
4. **Export Data:** CSV files with reproducibility metadata (seed included)
5. **Visualizations:** Charts and plots with stability sweep analysis

**Sample Output:**
```
Pick 5 (Round 1): Draft RB 
  Delta: +12.4 vs WR | DP Value: 1,234.5 | Best RB: P=0.95 available
Pick 24 (Round 2): Draft WR
  Delta: +8.1 vs RB | DP Value: 987.3 | Best WR: P=0.78 available  
Pick 33 (Round 3): Draft WR
  Delta: +5.7 vs RB | DP Value: 823.1 | Best WR: P=0.65 available
...
```
