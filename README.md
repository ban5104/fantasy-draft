# Fantasy Football Draft Optimizer

A data-driven draft optimization tool that uses dynamic programming and Monte Carlo simulation to determine optimal fantasy football draft strategies. Instead of evaluating individual players, this system uses a novel "Dynamic Programming over Positions" approach to maximize expected fantasy points across all draft picks.

**Recent Major Enhancements**: Comprehensive decision support system with ε-optimal draft strategies, regret analysis, flexibility scoring, contingency planning, and risk-adjusted variants. Enhanced analytics capture system with detailed pick analysis exports. Configurable draft position and league size with automatic snake pick calculation.

## Why This Approach

Traditional draft tools focus on player rankings and ADP (Average Draft Position). This optimizer goes deeper by:

- **Modeling draft randomness** with Monte Carlo simulation to predict player availability
- **Optimizing position selection** rather than specific players at each pick
- **Using dynamic programming** to find globally optimal strategies, not greedy pick-by-pick decisions
- **Accounting for roster construction** with position limits and snake draft dynamics
- **Providing comprehensive decision support** with multiple strategies, regret analysis, and contingency planning

The result is a draft-day command center that provides optimal position targeting, alternative strategies, and real-time decision support to maximize your team's expected point total.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate draft day cheat sheet with latest data
python scripts/update_cheat_sheet.py

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

### Draft Cheat Sheet

Generate a comprehensive cheat sheet that merges data from 4 sources:

```bash
# Update cheat sheet with all latest data
python scripts/update_cheat_sheet.py

# Verbose output to see data merge process  
python scripts/update_cheat_sheet.py -v
```

Creates `data/draft_day_cheat_sheet.csv` with columns:
- ESPN_ALG, ESPN_PROJ, ADP_SLEEPER, ACTUAL_DRAFT_YYYYMMDD
- Automatically finds latest files by timestamp
- 91.2% data coverage across all sources

### Mode Presets (Recommended)

```bash
# Fast mode - quick results for testing (100 simulations)
python scripts/dp_draft_optimizer_debug.py --mode fast

# Stable mode - production quality with decision support features (5000 simulations)
python scripts/dp_draft_optimizer_debug.py --mode stable

# Debug mode - full analysis with visualizations and enhanced features (1000 simulations)
python scripts/dp_draft_optimizer_debug.py --mode debug
```

All modes now include the comprehensive decision support features:
- **ε-Optimal Plans Menu**: Multiple draft strategies within 3.3% of optimal (configurable)
- **Pick Analysis**: Regret tables, flexibility scoring, contingency planning
- **Time-to-Cliff Warnings**: Value drop alerts for positional scarcity
- **K-Best Path Exploration**: Comprehensive alternative strategy discovery

### Envelope Projections

Use envelope projections to incorporate uncertainty ranges (floor/ceiling estimates) into your analysis:

```bash
# Enable envelope projections with CSV file
python scripts/dp_draft_optimizer_debug.py --mode stable --envelope-file data/my_envelope_projections.csv

# Enable comprehensive analytics data capture
python scripts/dp_draft_optimizer_debug.py --mode stable --capture-analytics

# Export analytics in Parquet format (more efficient for large datasets)
python scripts/dp_draft_optimizer_debug.py --mode stable --capture-analytics --export-parquet
```

### Advanced Usage

```bash
# Reproducible results with seed parameter
python scripts/dp_draft_optimizer_debug.py --mode stable --seed 42

# Parameter robustness testing
python scripts/dp_draft_optimizer_debug.py --stability-sweep

# Custom parameters (overrides mode presets)
python scripts/dp_draft_optimizer_debug.py --sims 10000 --randomness 0.4 --pool-size 20

# Custom ESPN data source for model comparison
python scripts/dp_draft_optimizer_debug.py --mode stable --espn-file data/espn_algorithm_20250824.csv --data-source algorithm

# Full feature set with envelope projections
python scripts/dp_draft_optimizer_debug.py --export-csv --export-simulations --visualize --save-plots --envelope-file data/projections.csv
```

### Complete Command Reference

```bash
# Core options
--mode {fast,stable,debug}          # Preset configurations (recommended)
--sims N                           # Number of Monte Carlo simulations
--seed N                           # Random seed for reproducibility

# Visualization and export
--visualize                        # Generate Monte Carlo dashboard
--save-plots                       # Save visualization plots as PNG
--export-csv                       # Export Monte Carlo results to CSV
--export-simulations               # Export individual simulation data
--enhanced-stats                   # Export statistical distributions

# Analytics and envelope projections
--capture-analytics                # Enable detailed analytics capture
--export-parquet                   # Export analytics in Parquet format
--envelope-file PATH               # Enable envelope projections

# Data and parameters
--espn-file PATH                   # Custom ESPN projections file
--data-source NAME                 # Data source identifier for outputs
--randomness FLOAT                 # Draft unpredictability (0.0-1.0)
--pool-size INT                    # Candidate pool size per pick

# Testing and analysis
--stability-sweep                  # Parameter robustness testing
--debug                           # Enable debug mode (default: true)
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
# Draft configuration (automatically calculates SNAKE_PICKS)
DRAFT_POSITION = 5          # Your position in the draft (1-14)
LEAGUE_SIZE = 14            # Number of teams in league

# Roster construction targets
POSITION_LIMITS = {'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1}

# Simulation parameters
RANDOMNESS_LEVEL = 0.3      # Draft unpredictability (0.1-0.7)
CANDIDATE_POOL_SIZE = 25    # Players each team considers per pick

# Enhanced decision support parameters
EPSILON_THRESHOLD = 0.033   # Show strategies within 3.3% of optimal
K_BEST_DEPTH = 10          # Alternative paths to explore per pick
```

## Data Requirements

The system uses flexible data loading with hierarchical matching:

- **ESPN Projections**: Any CSV file with columns: `player_name`, `position`, `overall_rank`
  - Configurable via `--espn-file` parameter  
  - Available data: `data/probability-models-draft/espn_projections_20250814.csv`, `espn_algorithm_20250824.csv`
- **Fantasy Rankings**: `data/rankings_top300_20250814.csv` with columns: `PLAYER`, `FANTASY_PTS`
- **Draft Cheat Sheet Sources**: 4 files in `data/probability-models-draft/` for comprehensive analysis
  - `espn_algorithm_*.csv` - ESPN algorithm rankings
  - `espn_projections_*.csv` - ESPN projected rankings
  - `realtime_adp_*.csv` - Sleeper Average Draft Position data
  - `actual_draft_results_*.csv` - Actual draft results from real leagues
  - All use standardized format: `overall_rank,position,position_rank,player_name,team`
  - Full player names (e.g., "Ja'Marr Chase" not "J. Chase") for accurate matching
- **Envelope Projections**: Optional CSV with uncertainty ranges (floor/ceiling estimates)
  - Required columns: player name, position, and projection range columns
  - Supported column names:
    - Name: `name`, `player`, `player_name`
    - Position: `pos`, `position`
    - Low: `low`, `floor`, `p10`
    - Projection: `proj`, `projection`, `mode`, `median`, `p50`, `center`
    - High: `high`, `ceiling`, `p90`

**Data Quality**: 99.6% exact matches with 4-tier hierarchical matching system and 92% fuzzy threshold.

## File Structure

```
dpscript-insights/
├── scripts/
│   ├── dp_draft_optimizer_debug.py    # Main optimizer (self-contained)
│   ├── update_cheat_sheet.py          # Draft cheat sheet generator
│   └── tests/                         # Unit and integration tests
├── data/
│   ├── probability-models-draft/      # ESPN projections, ADP, draft results
│   ├── output-simulations/            # Analytics exports and Monte Carlo results
│   ├── rankings_top300_20250814.csv   # Fantasy point rankings
│   └── draft_day_cheat_sheet.csv      # Generated cheat sheet (merged data)
├── jupyter-notebooks/                 # Analysis and visualization (3 notebooks)
├── tests/golden/                      # Golden master regression tests
├── specs/                             # Mathematical theory and planning
└── requirements.txt                   # Python dependencies
```

## Developer Guide

For implementation details, mathematical background, and development workflows, see [CLAUDE.md](CLAUDE.md).

For theoretical foundation and high-level strategy planning, see the `specs/` directory.

## Enhanced Decision Support Features

The optimizer now provides comprehensive draft-day analysis:

### Multiple Draft Strategies
- **ε-Optimal Plans Menu**: Shows multiple strategies within 3.3% of optimal EV (configurable, e.g., "Plan A: RB-QB-RB-WR... EV 1380.9")
- **Risk-Adjusted Variants**: Floor-focused (conservative) vs upside-focused (aggressive) approaches when envelope data available
- **K-Best Exploration**: Configurable depth for discovering alternative draft paths

### Per-Pick Analysis
- **Regret Analysis**: Compares alternative position choices with specific regret percentages
- **Flexibility Index**: Entropy-based scoring (0.0-1.0) quantifying decision flexibility
- **Time-to-Cliff Warnings**: Alerts when positions face significant value drops in coming picks
- **Contingency Playbooks**: Primary/secondary/tertiary recommendations with specific player targets

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

## Standard Output

The optimizer also produces:

1. **Optimal Strategy:** Which position to draft at each of your picks
2. **Enhanced Debug Analysis:** Both immediate decision value (Delta) and total DP value with counterfactual analysis
3. **Player Availability:** Survival probabilities with clear availability displays (e.g., "P=0.82 best-available")
4. **Export Data:** CSV files with reproducibility metadata (seed included)
5. **Visualizations:** Charts and plots with stability sweep analysis
6. **Analytics Data Exports:** Comprehensive data exports for further analysis when `--capture-analytics` is enabled

### Analytics Exports

When analytics capture is enabled (via `--capture-analytics` or envelope projections), the system exports detailed analysis data:

- **`pick_candidates.csv`**: Top candidates considered at each pick with envelope metrics
  - Includes: player name, projected points, availability probability, floor/ceiling values, safety/volatility indices
- **`value_decay.csv`**: Value dropoff analysis between picks by position
  - Shows absolute and percentage point drops between consecutive picks
- **`pos_outlook.csv`**: Positional outlook and availability trends
  - Position-level analysis across draft rounds
- **`run_metadata.json`**: Execution metadata for reproducibility
  - Timestamp, system info, input file hashes, exported file paths

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
