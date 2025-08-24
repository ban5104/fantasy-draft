#!/usr/bin/env python3
"""
DP Draft Optimizer with DEBUG MODE - shows detailed calculations.
"""

import argparse
import logging
import os
import sys
from functools import lru_cache
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from datetime import datetime

try:
    from visualize_mc import create_simple_dashboard
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Removed data_adapters import - using simplified file parameter system


# ==============================================================================
# CORE DATA STRUCTURES - Player class and utility functions
# ==============================================================================

class Player(NamedTuple):
    """
    Represents a fantasy football player with their key attributes.
    
    This is the core data structure used throughout the optimization system.
    """
    name: str
    position: str
    points: float
    overall_rank: int
    
    @property
    def unique_id(self) -> str:
        """Generate unique identifier from name and position to handle duplicate names."""
        return f"{self.name}_{self.position}"


def calculate_snake_picks(position: int, league_size: int = 14, rounds: int = 7) -> List[int]:
    """Calculate snake draft picks for given position."""
    picks = []
    for round_num in range(1, rounds + 1):
        if round_num % 2 == 1:  # Odd rounds go forward
            pick = (round_num - 1) * league_size + position
        else:  # Even rounds go backward
            pick = round_num * league_size - position + 1
        picks.append(pick)
    return picks


def strip_team_abbrev(name: str) -> str:
    """
    Remove team abbreviation from player name.
    
    Examples:
        'Josh Allen BUF' -> 'Josh Allen'
        'Christian McCaffrey SF' -> 'Christian McCaffrey'
        'Justin Jefferson' -> 'Justin Jefferson' (no change)
    
    Args:
        name: Player name potentially with team abbreviation
        
    Returns:
        Player name without team abbreviation
    """
    # Handle NaN or non-string values
    if not isinstance(name, str):
        return str(name) if name is not None else ""
    
    # Common team abbreviations are 2-3 uppercase letters at the end
    parts = name.rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isupper() and 2 <= len(parts[1]) <= 3:
        return parts[0]
    return name


def pos_sorted(players: list, position: str) -> list:
    """
    Return players of given position sorted by fantasy points (descending).
    
    This is a critical function for the ladder EV calculations - players
    must be sorted by fantasy points, not by ESPN rank.
    
    Args:
        players: List of Player objects
        position: Position to filter for (e.g., 'RB', 'WR', 'QB', 'TE')
        
    Returns:
        List of Player objects filtered by position and sorted by points
    """
    return sorted(
        (p for p in players if p.position == position),
        key=lambda p: p.points,
        reverse=True,
    )

# ==============================================================================
# CONFIGURATION - Adjust these parameters to control the simulation
# ==============================================================================

# Draft configuration - will be updated based on --position argument
DRAFT_POSITION = 5  # Default position, overridden by --position flag
LEAGUE_SIZE = 14    # 14-team league
TOTAL_ROUNDS = 14   # Total rounds to draft
SNAKE_PICKS = []    # Calculated automatically from position

# Maximum roster slots you need to fill (for first 7 rounds optimization)
POSITION_LIMITS = {"RB": 3, "WR": 2, "QB": 1, "TE": 1}  # Optimizing first 7 picks only

# ======= Envelope + Export Add-Ons (non-invasive) =======
# Put near the top of your file (after numpy/pandas imports)

ENVELOPE_FILE = ENVELOPE_FILE if "ENVELOPE_FILE" in globals() else None   # set a path or leave None
USE_ENVELOPES = True
EXPORT_DIR = "data/output-simulations"
EXPORT_FORMAT = "parquet"  # falls back to csv automatically

import os, json, hashlib, time, sys, platform
import numpy as np, pandas as pd

def _canon(s: str) -> str:
    s = (s or "").lower()
    for tok in [" jr.", " jr", " sr.", " sr", " iii", " ii", ".", ",", "'", "\"", "(", ")", "[", "]"]:
        s = s.replace(tok, " ")
    return " ".join(s.split())

def _safe_export(df: pd.DataFrame, name: str) -> str:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    base = os.path.join(EXPORT_DIR, name)
    if EXPORT_FORMAT == "parquet":
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(base + ".parquet", index=False)
            return base + ".parquet"
        except Exception:
            pass
    df.to_csv(base + ".csv", index=False)
    return base + ".csv"

def _hash_file(path: str) -> str:
    if not path or not os.path.exists(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_envelopes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in cols: return cols[o]
        raise KeyError(f"Missing column: one of {opts}")
    name_col = cols.get("name") or cols.get("player") or cols.get("player_name")
    pos_col  = pick("pos","position")
    low_col  = pick("low","floor","p10")
    mid_col  = pick("proj","projection","mode","median","p50","center")
    high_col = pick("high","ceiling","p90")
    df = df.rename(columns={name_col:"name", pos_col:"pos", low_col:"low", mid_col:"proj", high_col:"high"})
    df = df[df["high"] >= df["low"]]
    df["name_key"] = df["name"].map(_canon)
    eq = df["high"] == df["low"]
    df.loc[eq, "high"] = df.loc[eq, "high"] + 1e-6
    return df[["name","pos","low","proj","high","name_key"]]

# Note: Envelope integration happens in load_and_merge_data() function

# ======= 2) Capture structures (append alongside your prints) =======
if "_capture_pick_candidates" not in globals():
    _capture_pick_candidates = []
    _capture_value_decay = []
    _capture_pos_outlook = []

def _lookup_env(name: str):
    if "players_df" not in globals(): return (np.nan, np.nan, np.nan)
    row = players_df.loc[players_df["name"]==name, ["low","proj","high"]]
    if row.empty: return (np.nan, np.nan, np.nan)
    r = row.iloc[0]
    return (float(r.get("low", np.nan)), float(r.get("proj", np.nan)), float(r.get("high", np.nan)))

def _add_env_metrics(row: dict):
    # adds floor/ceiling/safety/volatility using low/proj/high (if present)
    out = dict(row)
    low, proj, high = out.get("low"), out.get("proj"), out.get("high")
    if (low is None or not np.isfinite(low)) or (proj is None or not np.isfinite(proj)) or (high is None or not np.isfinite(high)):
        # try to fill from lookup
        low, proj, high = _lookup_env(out["name"])
        out["low"], out["proj"], out["high"] = low, proj, high
    if np.isfinite(low) and np.isfinite(proj) and np.isfinite(high) and (proj != 0.0):
        eps = 1e-6
        out["floor"] = low
        out["ceiling"] = high
        out["safety_idx"] = low / (abs(proj) + eps)
        out["ceiling_idx"] = high / (abs(proj) + eps)
        out["volatility_idx"] = (high - low) / (abs(proj) + eps)
    return out

# ======= 3) Call these next to your existing print blocks =======
def _record_candidates(pick_num: int, pos: str, rows: list[dict]):
    """
    rows items expected: {"name": str, "proj_pts": float, "avail": float}
    We augment with low/proj/high + derived envelope metrics if available.
    """
    for r in rows:
        rec = {
            "pick": int(pick_num),
            "pos": pos,
            "name": r["name"],
            "proj_pts": float(r["proj_pts"]),
            "avail": float(r.get("avail", np.nan)),
        }
        low, proj, high = _lookup_env(r["name"])
        rec.update({"low": low, "proj": proj, "high": high})
        rec = _add_env_metrics(rec)
        _capture_pick_candidates.append(rec)

def _record_value_decay(pick_num: int, pos: str, now_pts: float, next_pts: float, now_survival: float | None):
    drop_abs = float(now_pts - next_pts)
    drop_pct = float(0.0 if now_pts == 0 else 100.0 * drop_abs / now_pts)
    _capture_value_decay.append({
        "pick": int(pick_num), "pos": pos,
        "now_pts": float(now_pts), "next_pts": float(next_pts),
        "now_survival": float(now_survival) if now_survival is not None else np.nan,
        "drop_abs": drop_abs, "drop_pct": drop_pct
    })

def _record_pos_outlook(pick_num: int, window_label: str, pos: str, rows: list[dict]):
    """
    rows items: {"name": str, "rel_baseline": float, "survival": float}
    We back out an estimated baseline to compute floor/ceiling vs baseline too.
    """
    for r in rows:
        name = r["name"]; rb = float(r["rel_baseline"])
        low, proj, high = _lookup_env(name)
        base_est = (proj / rb) if (rb and np.isfinite(rb) and np.isfinite(proj) and rb != 0.0) else np.nan
        floor_rel = (low / base_est) if (np.isfinite(low) and np.isfinite(base_est) and base_est != 0.0) else np.nan
        ceil_rel  = (high / base_est) if (np.isfinite(high) and np.isfinite(base_est) and base_est != 0.0) else np.nan
        _capture_pos_outlook.append({
            "pick": int(pick_num), "window": window_label, "pos": pos, "name": name,
            "rel_baseline": rb, "survival": float(r.get("survival", np.nan)),
            "proj_pts": float(proj) if np.isfinite(proj) else np.nan,
            "low": float(low) if np.isfinite(low) else np.nan,
            "high": float(high) if np.isfinite(high) else np.nan,
            "floor_rel_baseline": float(floor_rel) if np.isfinite(floor_rel) else np.nan,
            "ceiling_rel_baseline": float(ceil_rel) if np.isfinite(ceil_rel) else np.nan
        })

# ======= 4) Export everything at the end of the run (or after each pick) =======
def _export_pick_run(meta_extra: dict | None = None):
    cand_df = pd.DataFrame(_capture_pick_candidates)
    decay_df = pd.DataFrame(_capture_value_decay)
    out_df   = pd.DataFrame(_capture_pos_outlook)

    where = {}
    if not cand_df.empty:  where["pick_candidates"] = _safe_export(cand_df, "pick_candidates")
    if not decay_df.empty: where["value_decay"]     = _safe_export(decay_df, "value_decay")
    if not out_df.empty:   where["pos_outlook"]     = _safe_export(out_df, "pos_outlook")

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "envelope_file": ENVELOPE_FILE if ENVELOPE_FILE else None,
            "envelope_hash": _hash_file(ENVELOPE_FILE) if ENVELOPE_FILE else ""
        },
        "exported": where
    }
    if meta_extra: meta.update(meta_extra)
    with open(os.path.join(EXPORT_DIR, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

# ==============================================================================
# MONTE CARLO SIMULATION PARAMETERS - Control draft variability
# ==============================================================================

# RANDOMNESS_LEVEL: How much randomness in each pick (0.0 to 1.0)
# - 0.1 = Very predictable, picks closely follow ESPN rankings
# - 0.3 = Moderate variance (DEFAULT - realistic draft chaos)
# - 0.5 = High variance, significant departures from rankings
# - 0.7 = Extreme variance, very unpredictable draft
RANDOMNESS_LEVEL = 0.3

# CANDIDATE_POOL_SIZE: How many top players teams consider for each pick
# - 5  = Very predictable, only top ranked players get picked
# - 10 = Somewhat predictable
# - 15 = Moderate flexibility (DEFAULT)
# - 20 = High flexibility, reaches become more common
# - 25 = Very flexible, significant reaches possible
CANDIDATE_POOL_SIZE = 25

# Note: This simulation uses pure ESPN rankings with noise
# Teams pick best available players without position-based adjustments


# Global data for DP optimization
PLAYERS = []
SURVIVAL_PROBS = {}

# Capture variables for calculation examples
CAPTURE_EXAMPLES = {
    'monte_carlo_pick': None,
    'ladder_ev_calculation': None,
    'dp_decision_comparison': None
}



# ==============================================================================
# DATA LOADING AND PROCESSING - Load ESPN projections and fantasy rankings
# ==============================================================================


def load_and_merge_data(espn_file: str = "data/probability-models-draft/espn_projections_20250814.csv") -> List[Player]:
    """Load ESPN projections and fantasy points, merge via fuzzy matching.
    
    Args:
        espn_file: Path to ESPN projections CSV file
        
    Returns:
        List of Player objects with merged data
    """
    # Validate required CSV files exist before attempting to load
    points_file = "data/rankings_top300_20250814.csv"

    if not os.path.exists(espn_file):
        logging.error(f"Required ESPN projections file not found: {espn_file}")
        print(
            "Please ensure the data directory contains the ESPN projections CSV file."
        )
        sys.exit(1)

    if not os.path.exists(points_file):
        logging.error(f"Required rankings file not found: {points_file}")
        print("Please ensure the data directory contains the rankings CSV file.")
        sys.exit(1)

    try:
        espn = pd.read_csv(espn_file)
        points = pd.read_csv(points_file)
    except Exception as e:
        logging.error(f"Failed to load CSV files: {e}")
        sys.exit(1)

    # Filter out D/ST and K positions before matching to reduce noise
    espn = espn[~espn["position"].isin(["K", "DST", "D/ST", "DEF"])]
    if "position" in points.columns:
        points = points[~points["position"].isin(["K", "DST", "D/ST", "DEF"])]

    # Create multi-level lookups
    points_by_name_pos = {}
    points_by_name = {}
    for _, row in points.iterrows():
        name = row["PLAYER"]
        pos = row["POSITION"]
        pts = row["FANTASY_PTS"]
        points_by_name_pos[(name, pos)] = pts
        points_by_name[name] = pts

    players = []
    unmatched_elite = []
    match_stats = {"exact_pos": 0, "exact_name": 0, "fuzzy": 0, "unmatched": 0}
    fuzzy_cache = {}  # Cache fuzzy matches within this run

    for _, row in espn.iterrows():
        espn_name = row["player_name"]
        espn_pos = row["position"]
        is_elite = row["overall_rank"] <= 50
        
        # PASS 1: Exact match on (name, position)
        if (espn_name, espn_pos) in points_by_name_pos:
            fantasy_points = points_by_name_pos[(espn_name, espn_pos)]
            match_stats["exact_pos"] += 1
        # PASS 2: Exact match on name only
        elif espn_name in points_by_name:
            fantasy_points = points_by_name[espn_name]
            match_stats["exact_name"] += 1
        # PASS 3: Check fuzzy cache
        elif espn_name in fuzzy_cache:
            matched_name, _ = fuzzy_cache[espn_name]
            fantasy_points = points_by_name[matched_name]
            match_stats["fuzzy"] += 1
        # PASS 4: Fuzzy match as last resort
        else:
            best_match = None
            best_score = 0
            for name in points_by_name.keys():
                score = fuzz.ratio(espn_name.lower(), name.lower())
                if score > best_score:
                    best_score = score
                    best_match = name
            
            if best_score >= 92 and best_match:  # Higher threshold
                fantasy_points = points_by_name[best_match]
                fuzzy_cache[espn_name] = (best_match, best_score)
                match_stats["fuzzy"] += 1
                if best_score < 95:
                    print(f"  Fuzzy match: '{espn_name}' -> '{best_match}' ({best_score}%)")
            else:
                fantasy_points = 0.0
                match_stats["unmatched"] += 1
                if is_elite:
                    unmatched_elite.append((row["overall_rank"], espn_name))

        players.append(
            Player(
                name=row["player_name"],
                position=row["position"],
                points=fantasy_points,
                overall_rank=row["overall_rank"],
            )
        )

    # Print match statistics
    total_matched = match_stats["exact_pos"] + match_stats["exact_name"] + match_stats["fuzzy"]
    print(f"\n=== Name Matching Statistics ===")
    print(f"Total players: {len(players)}")
    print(f"Matched: {total_matched} ({100*total_matched/len(players):.1f}%)")
    print(f"  - Exact (name + position): {match_stats['exact_pos']}")
    print(f"  - Exact (name only): {match_stats['exact_name']}")
    print(f"  - Fuzzy matches (>= 92%): {match_stats['fuzzy']}")
    print(f"Unmatched: {match_stats['unmatched']} ({100*match_stats['unmatched']/len(players):.1f}%)")
    
    # CRITICAL: Warn about unmatched elite players
    if unmatched_elite:
        print(f"\n⚠️  CRITICAL WARNING: {len(unmatched_elite)} elite players (top 50) have 0 fantasy points!")
        for rank, name in sorted(unmatched_elite)[:10]:
            print(f"  Rank #{rank}: {name}")
        print("\nThis will severely impact draft optimization accuracy!")
        print("Consider updating the data files or manually mapping these players.")
    
    # Additional validation: Check for position-level issues
    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players = [p for p in players if p.position == pos]
        zero_count = sum(1 for p in pos_players if p.points == 0.0)
        if zero_count > 0:
            elite_zeros = sum(1 for p in pos_players if p.points == 0.0 and p.overall_rank <= 50)
            pct = 100 * zero_count / len(pos_players) if pos_players else 0
            print(f"\n{pos}: {zero_count}/{len(pos_players)} ({pct:.1f}%) have 0 points", end="")
            if elite_zeros > 0:
                print(f" - INCLUDING {elite_zeros} TOP-50 PLAYERS!", end="")
            print()

    # Create players dataframe for envelope integration
    global players_df
    players_df = pd.DataFrame([{
        'name': p.name,
        'position': p.position, 
        'points': p.points,
        'overall_rank': p.overall_rank
    } for p in players])
    
    # Apply envelope integration from paste-in update
    try:
        if USE_ENVELOPES and ENVELOPE_FILE and os.path.exists(ENVELOPE_FILE):
            players_df["name_key"] = players_df["name"].map(_canon)
            env = _load_envelopes(ENVELOPE_FILE)
            players_df = players_df.merge(env[["name_key","low","proj","high"]], on="name_key", how="left")
            print(f"\nLoaded envelope projections from {ENVELOPE_FILE}: {len(env)} rows")
        else:
            # ensure columns exist for downstream joins
            for c in ("low","proj","high"):
                if c not in players_df.columns:
                    players_df[c] = np.nan
    except Exception as _e:
        print(f"[envelopes] skipped: {_e}")

    return sorted(players, key=lambda p: p.overall_rank)


def export_mc_results_to_csv(players, survival_probs, snake_picks, num_sims, data_source="espn", seed=None, enhanced_stats=False):
    """Export Monte Carlo results to CSV files for Jupyter analysis."""
    import pandas as pd

    # Get absolute path to output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "output-simulations")

    # Helper function to create player data
    def create_player_row(player):
        row = {
            "player_name": player.name,
            "position": player.position,
            "fantasy_points": player.points,
            "overall_rank": player.overall_rank,
        }
        
        if enhanced_stats:
            # Calculate statistical distributions for each pick
            for pick in snake_picks:
                data_key = (player.unique_id, pick)
                if data_key in survival_probs and survival_probs[data_key]:
                    survival_array = np.array(survival_probs[data_key])
                    
                    # Basic statistics
                    mean_survival = np.mean(survival_array)
                    std_survival = np.std(survival_array, ddof=1) if len(survival_array) > 1 else 0.0
                    
                    # Percentiles
                    p5 = np.percentile(survival_array, 5)
                    p25 = np.percentile(survival_array, 25)
                    p75 = np.percentile(survival_array, 75)
                    p95 = np.percentile(survival_array, 95)
                    
                    # 95% Confidence intervals (assuming normal distribution)
                    if std_survival > 0 and len(survival_array) > 1:
                        from scipy import stats
                        ci_margin = stats.t.ppf(0.975, len(survival_array)-1) * (std_survival / np.sqrt(len(survival_array)))
                        ci_lower = max(0, mean_survival - ci_margin)
                        ci_upper = min(1, mean_survival + ci_margin)
                    else:
                        ci_lower = ci_upper = mean_survival
                    
                    # Update row with enhanced statistics
                    row.update({
                        f"survival_pick_{pick}_mean": mean_survival,
                        f"survival_pick_{pick}_std": std_survival,
                        f"survival_pick_{pick}_p25": p25,
                        f"survival_pick_{pick}_p75": p75,
                        f"survival_pick_{pick}_ci_lower": ci_lower,
                        f"survival_pick_{pick}_ci_upper": ci_upper,
                        f"survival_pick_{pick}_p5": p5,
                        f"survival_pick_{pick}_p95": p95,
                    })
                else:
                    # No data - set all stats to 0
                    row.update({
                        f"survival_pick_{pick}_mean": 0.0,
                        f"survival_pick_{pick}_std": 0.0,
                        f"survival_pick_{pick}_p25": 0.0,
                        f"survival_pick_{pick}_p75": 0.0,
                        f"survival_pick_{pick}_ci_lower": 0.0,
                        f"survival_pick_{pick}_ci_upper": 0.0,
                        f"survival_pick_{pick}_p5": 0.0,
                        f"survival_pick_{pick}_p95": 0.0,
                    })
        else:
            # Original behavior: just mean survival probabilities
            row.update(
                {
                    f"survival_pick_{pick}": survival_probs.get((player.unique_id, pick), 0.0)
                    for pick in snake_picks
                }
            )
        return row

    # 1. Player survival data
    player_data = [create_player_row(p) for p in players]
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(player_data).to_csv(
        os.path.join(output_dir, f"mc_player_survivals_{data_source}.csv"), index=False
    )

    # 2. Position summary data
    position_data = []
    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players = [p for p in players if p.position == pos][:10]
        for pick in snake_picks:
            if enhanced_stats:
                # Calculate statistics from survival data arrays
                survival_arrays = []
                for p in pos_players:
                    data_key = (p.unique_id, pick)
                    if data_key in survival_probs and survival_probs[data_key]:
                        survival_arrays.append(np.mean(survival_probs[data_key]))
                
                if survival_arrays:
                    position_data.append(
                        {
                            "position": pos,
                            "pick": pick,
                            "avg_survival_top10": np.mean(survival_arrays),
                            "max_survival_top10": np.max(survival_arrays),
                            "min_survival_top10": np.min(survival_arrays),
                        }
                    )
            else:
                # Original behavior: use survival probabilities directly
                survivals = [survival_probs.get((p.unique_id, pick), 0.0) for p in pos_players]
                if survivals:
                    position_data.append(
                        {
                            "position": pos,
                            "pick": pick,
                            "avg_survival_top10": np.mean(survivals),
                            "max_survival_top10": np.max(survivals),
                            "min_survival_top10": np.min(survivals),
                        }
                    )

    pd.DataFrame(position_data).to_csv(
        os.path.join(output_dir, f"mc_position_summary_{data_source}.csv"), index=False
    )

    # 3. Configuration metadata
    config_data = {
        "num_simulations": [num_sims],
        "snake_picks": [str(snake_picks)],
        "position_limits": [str(POSITION_LIMITS)],
        "export_timestamp": [datetime.now().isoformat()],
        "total_players": [len(players)],
        "randomness_level": [RANDOMNESS_LEVEL],
        "candidate_pool_size": [CANDIDATE_POOL_SIZE],
        "data_source": [data_source],
        "seed": [seed if seed is not None else "None"],
        "enhanced_stats": [enhanced_stats],
    }
    pd.DataFrame(config_data).to_csv(
        os.path.join(output_dir, f"mc_config_{data_source}.csv"), index=False
    )

    print(f"\nExported Monte Carlo results to CSV files ({data_source.upper()} data source):")
    print(
        f"  - {os.path.join(output_dir, f'mc_player_survivals_{data_source}.csv')} ({len(player_data)} players)"
    )
    print(
        f"  - {os.path.join(output_dir, f'mc_position_summary_{data_source}.csv')} ({len(position_data)} position/pick combinations)"
    )
    print(f"  - {os.path.join(output_dir, f'mc_config_{data_source}.csv')} (simulation metadata)")
    print(
        f"\nLoad in Jupyter with: pd.read_csv('{os.path.join(output_dir, f'mc_player_survivals_{data_source}.csv')}')"
    )


def _export_pick_run(meta_extra: dict | None = None):
    """Export captured analytics data with metadata exactly as specified in feedback."""
    cand_df = pd.DataFrame(_capture_pick_candidates)
    decay_df = pd.DataFrame(_capture_value_decay)
    out_df = pd.DataFrame(_capture_pos_outlook)

    where = {}
    if not cand_df.empty:  where["pick_candidates"] = _safe_export(cand_df, "pick_candidates")
    if not decay_df.empty: where["value_decay"]     = _safe_export(decay_df, "value_decay")
    if not out_df.empty:   where["pos_outlook"]     = _safe_export(out_df, "pos_outlook")

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "envelope_file": ENVELOPE_FILE,
            "envelope_hash": _hash_file(ENVELOPE_FILE),
        },
        "exported": where
    }
    if meta_extra: meta.update(meta_extra)
    
    # Export metadata
    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(os.path.join(EXPORT_DIR, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Print summary of exports
    if where:
        print(f"\nExported analytics data ({EXPORT_FORMAT.upper()} format):")
        for key, filepath in where.items():
            print(f"  - {key}: {filepath}")
        print(f"  - metadata: {os.path.join(EXPORT_DIR, 'run_metadata.json')}")
    else:
        print("\nNo analytics data captured (all capture lists were empty)")


def monte_carlo_survival_realistic(
    players: List[Player], num_sims: int, export_simulation_data: bool = False, data_source: str = "espn", enhanced_stats: bool = False
) -> Dict[Tuple[str, int], float]:
    """
    More realistic Monte Carlo with position scarcity modeling.
    Returns survival_probs[(player_unique_id, pick_number)] = probability
    
    If enhanced_stats=True, returns dict with statistical distributions instead of just means
    """
    if enhanced_stats:
        # Track survival data per simulation for statistical calculations
        survival_data = {(p.unique_id, pick): [] for p in players for pick in SNAKE_PICKS}
    else:
        # Original behavior: just count survivals
        survival_counts = {(p.unique_id, pick): 0 for p in players for pick in SNAKE_PICKS}
    
    simulation_picks = []

    for sim in range(num_sims):
        available_mask = np.ones(len(players), dtype=bool)
        available_indices = np.arange(len(players))

        for pick_num in range(1, 90):
            if not np.any(available_mask):
                break

            if pick_num in SNAKE_PICKS:
                if enhanced_stats:
                    # Track survival (1) or non-survival (0) for each player at this pick
                    for i, player in enumerate(players):
                        survival_data[(player.unique_id, pick_num)].append(1 if available_mask[i] else 0)
                else:
                    # Original behavior: just count survivals
                    for i, player in enumerate(players):
                        if available_mask[i]:
                            survival_counts[(player.unique_id, pick_num)] += 1
            else:
                # Dynamic candidate pool sizing based on pick timing
                if pick_num <= 30:
                    pool_size = CANDIDATE_POOL_SIZE
                elif pick_num <= 60:
                    pool_size = min(
                        np.sum(available_mask), int(CANDIDATE_POOL_SIZE * 1.5)
                    )
                else:
                    pool_size = min(
                        np.sum(available_mask), int(CANDIDATE_POOL_SIZE * 2)
                    )
                valid_indices = available_indices[available_mask][:pool_size]
                candidates = [players[i] for i in valid_indices]
                # Original Gaussian noise approach
                scores = [
                    (1.0 / (i + 1)) * max(0.1, np.random.normal(1.0, RANDOMNESS_LEVEL))
                    for i, p in enumerate(candidates)
                ]

                if scores:
                    # Normalize weights with epsilon to avoid division by zero
                    weights = np.array(scores)
                    weights = weights / (weights.sum() + 1e-12)
                    choice_idx = np.random.choice(len(candidates), p=weights)
                    picked_player = candidates[choice_idx]
                    player_idx = players.index(
                        picked_player
                    )  # Calculate once per picked player
                    available_mask[player_idx] = False
                    
                    # Capture ONE example for technical summary (sim 0, pick 15)
                    if sim == 0 and pick_num == 15 and CAPTURE_EXAMPLES['monte_carlo_pick'] is None:
                        CAPTURE_EXAMPLES['monte_carlo_pick'] = {
                            'sim': sim,
                            'pick_num': pick_num,
                            'candidates': [(c.name, c.position, c.overall_rank) for c in candidates[:5]],
                            'scores': scores[:5],
                            'weights': weights[:5].tolist(),
                            'choice_idx': choice_idx,
                            'picked_player': (picked_player.name, picked_player.position, picked_player.overall_rank),
                            'randomness_level': RANDOMNESS_LEVEL
                        }

                    if export_simulation_data:
                        simulation_picks.append(
                            {
                                "simulation": sim,
                                "pick_number": pick_num,
                                "player_name": picked_player.name,
                                "position": picked_player.position,
                                "espn_rank": picked_player.overall_rank,
                            }
                        )

    # Export simulation data if requested
    if export_simulation_data and simulation_picks:
        import pandas as pd

        # Get absolute path to output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "data", "output-simulations")
        pd.DataFrame(simulation_picks).to_csv(
            os.path.join(output_dir, f"mc_simulation_picks_{data_source}.csv"), index=False
        )
        print(
            f"Exported {len(simulation_picks)} individual simulation picks to {os.path.join(output_dir, f'mc_simulation_picks_{data_source}.csv')}"
        )

    if enhanced_stats:
        # Return survival data arrays for statistical processing
        return survival_data
    else:
        # Original behavior: return average survival probabilities
        return {key: count / num_sims for key, count in survival_counts.items()}


def get_position_survival_matrix(
    players: List[Player], survival_probs: Dict[Tuple[str, int], float]
) -> Dict[str, np.ndarray]:
    """Convert player-level survival to position matrices for ladder EV."""
    position_matrices = {}

    for pos in ["RB", "WR", "QB", "TE"]:
        # CRITICAL FIX: Use points-sorted order to match ladder calculations
        pos_players = pos_sorted(players, pos)

        # Build survival matrix for this position
        max_pick = max(SNAKE_PICKS) + 10
        matrix = np.zeros((len(pos_players), max_pick + 1))

        for i, player in enumerate(pos_players):
            for pick in SNAKE_PICKS:
                matrix[i, pick] = survival_probs.get((player.unique_id, pick), 0.0)

        position_matrices[pos] = matrix

        # Data validation - warn if all survival probabilities are 0
        if matrix.sum() == 0:
            print(f"WARNING: No survival probabilities found for position {pos}")
        else:
            avg_survival = matrix[matrix > 0].mean() if matrix.sum() > 0 else 0
            print(
                f"Position {pos}: {len(pos_players)} players, avg survival: {avg_survival:.3f}"
            )

    # Add sanity checks to catch misalignment
    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players_sorted = pos_sorted(players, pos)
        if pos in position_matrices:
            assert position_matrices[pos].shape[0] == len(
                pos_players_sorted
            ), f"Matrix rows ({position_matrices[pos].shape[0]}) != players ({len(pos_players_sorted)}) for {pos}"
            # MONOTONIC SURVIVAL CHECK: survival should never increase for later picks
            M = position_matrices[pos]
            snake_pick_indices = [
                i for i, pick in enumerate(range(M.shape[1])) if pick in SNAKE_PICKS
            ]
            if len(snake_pick_indices) > 1:
                # Extract just the snake pick columns and apply cumulative minimum
                snake_cols = M[:, snake_pick_indices]
                monotonic_cols = np.minimum.accumulate(snake_cols, axis=1)
                # Put the smoothed values back
                M[:, snake_pick_indices] = monotonic_cols
                position_matrices[pos] = M

    return position_matrices


def ladder_ev_debug(
    position: str,
    pick_number: int,
    slot: int,
    players: List[Player],
    survival_probs: Dict[str, np.ndarray],
) -> Tuple[float, List[str]]:
    """Compute expected value with debug info."""
    # CRITICAL FIX: Use points-sorted order to match survival matrix
    pos_players = pos_sorted(players, position)
    if not pos_players or position not in survival_probs:
        return 0.0, [f"  WARNING: No players or survival data for position {position}"]

    survival = survival_probs[position]
    debug_info = [f"  Position {position}, Slot {slot} at Pick {pick_number}:"]
    expected_value = 0.0

    # Validate bounds
    if survival.size == 0 or pick_number >= survival.shape[1]:
        debug_info.append(
            f"  WARNING: Invalid matrix dimensions or pick number for {position}"
        )
        return 0.0, debug_info
    # Sanity check - ensure pick_number is valid
    assert (
        0 <= pick_number < survival.shape[1]
    ), f"Invalid pick_number {pick_number} for matrix shape {survival.shape}"

    # Dynamic debug limit based on pick timing to show relevant players
    debug_limit = 25 if pick_number >= 60 else 15 if pick_number >= 30 else 10
    for j in range(slot - 1, min(len(pos_players), survival.shape[0], debug_limit)):
        if j < 0:
            continue

        player = pos_players[j]
        surv_prob = survival[j, pick_number] if j < survival.shape[0] else 0.0

        # Calculate probability all better players are taken
        taken_prob = 1.0
        for h in range(slot - 1, j):
            if h < survival.shape[0] and pick_number < survival.shape[1]:
                taken_prob *= 1 - survival[h, pick_number]
            else:
                taken_prob = 0.0
                break

        contribution = player.points * surv_prob * taken_prob
        expected_value += contribution

        if contribution > 0.01 or j < slot + 2:
            debug_info.append(
                f"    {player.name}: {player.points:.1f}pts × {surv_prob:.2f}surv × {taken_prob:.2f}gone = {contribution:.1f}"
            )

    debug_info.append(f"    Total EV: {expected_value:.1f}")
    
    # Capture ONE example for technical summary (first slot calculation at any major pick)
    if (slot == 1 and pick_number in [5, 24, 33, 52] and 
        CAPTURE_EXAMPLES['ladder_ev_calculation'] is None):
        # Get detailed calculation for first few players
        calculation_details = []
        for j in range(slot - 1, min(len(pos_players), survival.shape[0], 3)):
            if j < 0:
                continue
            player = pos_players[j]
            surv_prob = survival[j, pick_number] if j < survival.shape[0] else 0.0
            taken_prob = 1.0
            for h in range(slot - 1, j):
                if h < survival.shape[0] and pick_number < survival.shape[1]:
                    taken_prob *= 1 - survival[h, pick_number]
            contribution = player.points * surv_prob * taken_prob
            calculation_details.append({
                'player_name': player.name,
                'fantasy_points': player.points,
                'survival_prob': surv_prob,
                'prob_better_gone': taken_prob,
                'contribution': contribution
            })
        
        CAPTURE_EXAMPLES['ladder_ev_calculation'] = {
            'position': position,
            'pick_number': pick_number,
            'slot': slot,
            'formula': 'EV = Σ(player_points × survival_prob × prob_better_gone)',
            'calculation_details': calculation_details,
            'total_ev': expected_value
        }
    
    return expected_value, debug_info


def show_pick_analysis(pick_idx: int, pick_number: int, counts: Dict[str, int]):
    """Show detailed analysis for a specific pick."""

    print(f"\n{'='*60}")
    print(f"PICK {pick_number} ANALYSIS (Pick #{pick_idx + 1} of {len(SNAKE_PICKS)})")
    print(
        f"Current roster: RB={counts['RB']}, WR={counts['WR']}, QB={counts['QB']}, TE={counts['TE']}"
    )
    print(f"{'='*60}")

    # Calculate EV and DP values for each position
    position_evs = {}
    position_dp_values = {}
    for pos in ["RB", "WR", "QB", "TE"]:
        if counts[pos] < POSITION_LIMITS[pos]:
            slot = counts[pos] + 1
            ev, debug_info = ladder_ev_debug(
                pos, pick_number, slot, PLAYERS, SURVIVAL_PROBS
            )
            position_evs[pos] = ev
            # Calculate total DP value if we draft this position
            new_counts = counts.copy()
            new_counts[pos] += 1
            future_value, _ = dp_optimize(
                pick_idx + 1,
                new_counts["RB"],
                new_counts["WR"],
                new_counts["QB"],
                new_counts["TE"],
            )
            total_dp_value = ev + future_value
            position_dp_values[pos] = total_dp_value

            print(f"\n{pos} Analysis:")
            for line in debug_info:  # Show all debug lines
                print(line)

            # Calculate delta to next pick
            if pick_idx + 1 < len(SNAKE_PICKS):
                next_pick = SNAKE_PICKS[pick_idx + 1]
                next_ev, _ = ladder_ev_debug(
                    pos, next_pick, slot, PLAYERS, SURVIVAL_PROBS
                )
                delta = ev - next_ev
                print(f"  Delta (now vs pick {next_pick}): {delta:.1f}")
            # Show total DP value
            print(f"  Total DP Value (V_P): {total_dp_value:.1f}")
        else:
            # Position is full
            position_dp_values[pos] = -float("inf")

    # Show counterfactual DP values summary
    print(f"\nCounterfactual DP Values Summary:")
    for pos in ["RB", "WR", "QB", "TE"]:
        if counts[pos] < POSITION_LIMITS[pos]:
            print(f"  {pos}: V_P = {position_dp_values[pos]:.1f}")
        else:
            print(f"  {pos}: FULL (cannot draft)")

    return position_evs


@lru_cache(maxsize=None)
def dp_optimize(
    pick_idx: int, rb_count: int, wr_count: int, qb_count: int, te_count: int
) -> Tuple[float, str]:
    """DP recurrence: F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}"""

    # Early termination if roster is full
    if rb_count == 3 and wr_count == 2 and qb_count == 1 and te_count == 1:
        return 0.0, ""

    if pick_idx >= len(SNAKE_PICKS):
        return 0.0, ""

    # Basic validation
    if pick_idx < 0 or not PLAYERS or not SURVIVAL_PROBS:
        return 0.0, ""

    current_pick = SNAKE_PICKS[pick_idx]
    counts = {"RB": rb_count, "WR": wr_count, "QB": qb_count, "TE": te_count}

    # Validate position counts
    if any(
        count < 0 or count > POSITION_LIMITS.get(pos, 0)
        for pos, count in counts.items()
    ):
        return 0.0, ""

    best_value = -float("inf")
    best_position = ""
    position_values = {}

    for pos in ["RB", "WR", "QB", "TE"]:
        if counts[pos] < POSITION_LIMITS[pos]:
            slot = counts[pos] + 1

            try:
                ev, debug_lines = ladder_ev_debug(
                    pos, current_pick, slot, PLAYERS, SURVIVAL_PROBS
                )

                new_counts = counts.copy()
                new_counts[pos] += 1

                future_value, _ = dp_optimize(
                    pick_idx + 1,
                    new_counts["RB"],
                    new_counts["WR"],
                    new_counts["QB"],
                    new_counts["TE"],
                )

                total_value = ev + future_value
                position_values[pos] = {'ev': ev, 'future_value': future_value, 'total_value': total_value}

                if total_value > best_value:
                    best_value = total_value
                    best_position = pos

            except Exception as e:
                print(
                    f"ERROR: Exception in DP optimization for {pos} at pick {current_pick}: {e}"
                )
                continue
    
    # Capture ONE example for technical summary (any major pick decision)
    if (current_pick in [5, 24, 33, 52] and 
        CAPTURE_EXAMPLES['dp_decision_comparison'] is None and position_values):
        CAPTURE_EXAMPLES['dp_decision_comparison'] = {
            'pick_number': current_pick,
            'pick_idx': pick_idx,
            'current_counts': counts.copy(),
            'position_values': position_values.copy(),
            'best_position': best_position,
            'best_value': best_value,
            'formula': 'DP_Value = Ladder_EV + Future_Value'
        }

    return (best_value, best_position) if best_value != -float("inf") else (0.0, "")


def run_stability_sweep(players: List[Player]) -> None:
    """Run parameter stability sweep to test robustness."""
    global RANDOMNESS_LEVEL, CANDIDATE_POOL_SIZE

    print("\n" + "=" * 60)
    print("PARAMETER STABILITY SWEEP")
    print("=" * 60)

    # Store original values
    orig_randomness = RANDOMNESS_LEVEL
    orig_pool_size = CANDIDATE_POOL_SIZE

    # Test different parameter combinations
    randomness_levels = [0.2, 0.3, 0.4]
    pool_sizes = [15, 20, 25]

    results = []

    for rand_level in randomness_levels:
        for pool_size in pool_sizes:
            RANDOMNESS_LEVEL = rand_level
            CANDIDATE_POOL_SIZE = pool_size

            # Run shorter simulation for sweep
            player_survival = monte_carlo_survival_realistic(players, 1000, data_source="sweep")
            SURVIVAL_PROBS = get_position_survival_matrix(players, player_survival)

            # Get optimal sequence
            sequence = []
            counts = {"RB": 0, "WR": 0, "QB": 0, "TE": 0}

            dp_optimize.cache_clear()
            for pick_idx in range(len(SNAKE_PICKS)):
                value, position = dp_optimize(
                    pick_idx, counts["RB"], counts["WR"], counts["QB"], counts["TE"]
                )
                sequence.append(position)
                if position:
                    counts[position] += 1

            # Calculate total expected value
            total_value, _ = dp_optimize(0, 0, 0, 0, 0)

            results.append(
                {
                    "randomness": rand_level,
                    "pool_size": pool_size,
                    "sequence": "-".join(sequence),
                    "total_value": total_value,
                }
            )

            print(
                f"  Rand={rand_level}, Pool={pool_size}: {'-'.join(sequence)} (EV={total_value:.1f})"
            )
    # Restore original values
    RANDOMNESS_LEVEL = orig_randomness
    CANDIDATE_POOL_SIZE = orig_pool_size

    # Analyze stability
    unique_sequences = set(r["sequence"] for r in results)
    print(f"\nStability Analysis:")
    print(
        f"  Unique sequences found: {len(unique_sequences)} out of {len(results)} parameter combinations"
    )

    if len(unique_sequences) == 1:
        print("  Result: HIGHLY STABLE - Same sequence across all parameters")
    elif len(unique_sequences) <= 3:
        print("  Result: STABLE - Few sequence variations")
    else:
        print("  Result: UNSTABLE - Many sequence variations")
    # Show most common sequence
    from collections import Counter

    sequence_counts = Counter(r["sequence"] for r in results)
    most_common = sequence_counts.most_common(1)[0]
    print(
        f"  Most common sequence: {most_common[0]} (appeared {most_common[1]}/{len(results)} times)"
    )


def show_technical_summary():
    """Display position stats, top contributors, and calculation examples."""
    print("\n" + "=" * 60)
    print("TECHNICAL SUMMARY")
    print("=" * 60)
    
    if not PLAYERS or not SURVIVAL_PROBS:
        print("WARNING: No players or survival data available")
        return
    
    # Position stats and top contributors
    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players = pos_sorted(PLAYERS, pos)
        matrix = SURVIVAL_PROBS.get(pos)
        
        print(f"{pos}: {len(pos_players)} players loaded")
        if matrix is not None and matrix.size > 0:
            avg_survival = matrix[matrix > 0].mean() if matrix.sum() > 0 else 0
            print(f"  Matrix: {matrix.shape[0]}x{matrix.shape[1]}, avg survival: {avg_survival:.3f}")
            print(f"  Top contributors:")
            
            for i, player in enumerate(pos_players[:3]):
                if i < matrix.shape[0] and SNAKE_PICKS[0] < matrix.shape[1]:
                    first_pick_surv = matrix[i, SNAKE_PICKS[0]]
                    print(f"    {player.name[:25]:25} {player.points:.0f}pts ({first_pick_surv:.0%} P{SNAKE_PICKS[0]})")
    
    # Calculation examples section
    print(f"\n{'='*60}")
    print("ALGORITHM CALCULATION EXAMPLES")
    print(f"{'='*60}")
    print("The following examples show how the algorithm works with actual numbers from this run.")
    
    # 1. Monte Carlo Pick Example
    mc_example = CAPTURE_EXAMPLES.get('monte_carlo_pick')
    if mc_example:
        print(f"\n1. MONTE CARLO PICK EXAMPLE (Sim {mc_example['sim']}, Pick {mc_example['pick_num']}):")
        print(f"   How the algorithm simulates other teams' draft picks:")
        print(f"   Randomness Level: {mc_example['randomness_level']} (affects selection variance)")
        print(f"   Available Candidate Pool (top 5 shown):")
        for i, (name, pos, rank) in enumerate(mc_example['candidates']):
            score = mc_example['scores'][i]
            weight = mc_example['weights'][i]
            selected = "← SELECTED" if i == mc_example['choice_idx'] else ""
            print(f"     {name[:20]:20} {pos:2} (ESPN #{rank:2}) score:{score:.3f} weight:{weight:.3f} {selected}")
        picked_name, picked_pos, picked_rank = mc_example['picked_player']
        print(f"   Result: {picked_name} ({picked_pos}, ESPN #{picked_rank}) was selected")
        print(f"   Formula: score = (1/espn_rank) × max(0.1, normal(1.0, {mc_example['randomness_level']}))")
        print(f"            weights = scores / sum(scores)")
        print(f"   This process repeats {mc_example['sim']+1000} times to calculate survival probabilities.")
    else:
        print(f"\n1. MONTE CARLO PICK EXAMPLE: Not captured")
    
    # 2. Ladder EV Calculation Example
    ladder_example = CAPTURE_EXAMPLES.get('ladder_ev_calculation')
    if ladder_example:
        print(f"\n2. LADDER EV CALCULATION EXAMPLE:")
        print(f"   How the algorithm calculates expected value for drafting a position:")
        print(f"   Position: {ladder_example['position']}, Pick: {ladder_example['pick_number']}, Slot: {ladder_example['slot']} (your {ladder_example['slot']}{'st' if ladder_example['slot']==1 else 'nd' if ladder_example['slot']==2 else 'rd' if ladder_example['slot']==3 else 'th'} {ladder_example['position']})")
        print(f"   Formula: {ladder_example['formula']}")
        print(f"   Detailed Calculation:")
        for detail in ladder_example['calculation_details']:
            print(f"     {detail['player_name'][:20]:20} {detail['fantasy_points']:3.0f}pts × {detail['survival_prob']:.2f}surv × {detail['prob_better_gone']:.2f}gone = {detail['contribution']:4.1f}")
        print(f"   Total Expected Value: {ladder_example['total_ev']:.1f} fantasy points")
        print(f"   Interpretation: If you draft {ladder_example['position']} at pick {ladder_example['pick_number']}, you expect {ladder_example['total_ev']:.1f} points.")
    else:
        print(f"\n2. LADDER EV CALCULATION EXAMPLE: Not captured")
    
    # 3. DP Decision Comparison Example
    dp_example = CAPTURE_EXAMPLES.get('dp_decision_comparison')
    if dp_example:
        print(f"\n3. DP DECISION COMPARISON EXAMPLE (Pick {dp_example['pick_number']}):")
        print(f"   How the algorithm chooses the optimal position to draft:")
        print(f"   Current Roster: RB={dp_example['current_counts']['RB']}/{POSITION_LIMITS['RB']}, WR={dp_example['current_counts']['WR']}/{POSITION_LIMITS['WR']}, QB={dp_example['current_counts']['QB']}/{POSITION_LIMITS['QB']}, TE={dp_example['current_counts']['TE']}/{POSITION_LIMITS['TE']}")
        print(f"   Formula: {dp_example['formula']}")
        print(f"   Position Comparison:")
        print(f"     Position   Ladder_EV  +  Future_Value  =  Total_Value")
        for pos in ["RB", "WR", "QB", "TE"]:
            if pos in dp_example['position_values']:
                values = dp_example['position_values'][pos]
                winner = "← WINNER" if pos == dp_example['best_position'] else ""
                print(f"        {pos}:     {values['ev']:6.1f}  +     {values['future_value']:6.1f}  =     {values['total_value']:6.1f} {winner}")
            else:
                print(f"        {pos}:       FULL (roster position limit reached)")
        print(f"   Decision: Draft {dp_example['best_position']} (maximizes Total Value: {dp_example['best_value']:.1f})")
        print(f"   Interpretation: {dp_example['best_position']} gives the highest combined value from this pick plus all future picks.")
    else:
        print(f"\n3. DP DECISION COMPARISON EXAMPLE: Not captured")
    
    # 4. Key Formulas Reference
    print(f"\n4. KEY FORMULAS REFERENCE:")
    print(f"   How each component is calculated:")
    print(f"   • Survival Probability: P(player still available at pick) from Monte Carlo simulation")
    print(f"   • Prob Better Gone: ∏(1 - survival_prob_of_better_players) - all better players taken")
    print(f"   • Ladder EV: Σ(fantasy_points × survival_prob × prob_better_gone) - expected points")
    print(f"   • DP Recursion: F(pick,roster) = max_position(ladder_ev + F(next_pick,updated_roster))")
    print(f"   • Monte Carlo Weight: (1/espn_rank) × max(0.1, normal(1.0, randomness)) - selection probability")
    print(f"\n   Algorithm Flow:")
    print(f"   1. Simulate 1000+ drafts → Calculate survival probabilities")
    print(f"   2. For each pick/position → Calculate ladder expected value")
    print(f"   3. Use dynamic programming → Find optimal sequence maximizing total points")
    
    # Show interpretation help
    print(f"\n" + "─" * 60)
    print("INTERPRETATION GUIDE:")
    print("• Survival Prob: Higher = more likely player available at that pick")
    print("• Prob Better Gone: Higher = more likely all better players are taken")
    print("• Ladder EV: Expected fantasy points if you draft this position now")
    print("• Future Value: Expected points from all remaining picks if you draft this position")
    print("• Total DP Value: Ladder EV + Future Value (what the algorithm maximizes)")
    print("─" * 60)


def show_top_players_survival():
    """Show survival probabilities for top players at each position."""

    print("\n" + "=" * 60)
    print("TOP PLAYER SURVIVAL PROBABILITIES")
    print("=" * 60)

    if not PLAYERS or not SURVIVAL_PROBS:
        print("WARNING: No players or survival data available")
        return

    for pos in ["RB", "WR", "QB", "TE"]:
        print(f"\n{pos} Top 5:")
        # CRITICAL FIX: Use points-sorted order to match ladder calculations
        pos_players = pos_sorted(PLAYERS, pos)[:5]

        if pos not in SURVIVAL_PROBS or SURVIVAL_PROBS[pos].size == 0:
            print(f"  WARNING: No survival data for position {pos}")
            continue

        survival_matrix = SURVIVAL_PROBS[pos]

        for i, player in enumerate(pos_players):
            if i >= survival_matrix.shape[0]:
                break

            surv_str = " ".join(
                [
                    (
                        f"P{pick}:{survival_matrix[i, pick]:.0%}"
                        if pick < survival_matrix.shape[1]
                        else f"P{pick}:N/A"
                    )
                    for pick in SNAKE_PICKS
                ]
            )
            print(f"  {player.name[:20]:20} ({player.points:.0f}pts) - {surv_str}")


def get_available_players(pos: str, pick_number: int, favorites: set) -> List[Tuple]:
    """Get available players for position with survival rates."""
    pos_players = pos_sorted(PLAYERS, pos)
    survival_matrix = SURVIVAL_PROBS.get(pos)
    
    if survival_matrix is None or survival_matrix.size == 0:
        return []
    
    available = []
    for i, player in enumerate(pos_players):
        if i < survival_matrix.shape[0] and pick_number < survival_matrix.shape[1]:
            survival_prob = survival_matrix[i, pick_number]
            if survival_prob > 0.05:  # Only show players with >5% survival
                available.append((player, survival_prob, player.name in favorites))
    return available


def show_position_players(pos: str, recommended_pos: str, available_players: List[Tuple], pick_number: int = None):
    """Display players for a position."""
    if not available_players:
        return
    
    limit = 7 if pos == recommended_pos else 3
    top_players = available_players[:limit]
    
    label = f"{pos.upper()} - RECOMMENDED:" if pos == recommended_pos else f"{pos.lower()} - also available:"
    print(f"{label}")
    
    # Only show headers for the recommended position
    if pos == recommended_pos:
        print()
        print(f"{'PLAYER':<25} {'POINTS':>6} {'AVAIL%':>8}")
        print("─" * 45)
    print()
    
    # Prepare data for recording if we're in the recommended position
    candidate_rows = []
    
    for player, survival_prob, is_favorite in top_players:
        marker = " ⭐" if is_favorite else ""
        print(f"{player.name[:25]:25} {player.points:3.0f} pts    {survival_prob:3.0%}{marker}")
        
        # Prepare for recording
        if pos == recommended_pos and pick_number is not None:
            candidate_rows.append({
                "name": player.name,
                "proj_pts": player.points,
                "avail": survival_prob
            })
    
    # Record candidates if this is the recommended position and we have analytics enabled
    if USE_ENVELOPES and pos == recommended_pos and pick_number is not None and candidate_rows:
        _record_candidates(pick_number, pos, candidate_rows)
    
    print()


def show_positional_outlook(pos: str, counts: Dict[str, int], pick_number: int, favorites: set):
    """Show outlook for a position - current best available and future picks."""
    
    pos_players = pos_sorted(PLAYERS, pos)
    survival_matrix = SURVIVAL_PROBS.get(pos)
    
    if survival_matrix is None or survival_matrix.size == 0:
        return
    
    # Determine baseline for percentage calculations
    if counts[pos] < POSITION_LIMITS[pos]:
        # Still need more players - use best AVAILABLE player (with ≥50% survival) as 100%
        baseline_player = None
        # Find best available player with ≥50% survival to use as baseline
        for i, player in enumerate(pos_players):
            if i < survival_matrix.shape[0] and pick_number < survival_matrix.shape[1]:
                surv = survival_matrix[i, pick_number]
                if surv >= 0.5:
                    baseline_player = player
                    break
        # Fallback to best overall if no one has ≥50% survival
        if baseline_player is None and len(pos_players) > 0:
            baseline_player = pos_players[0]
        
        # More specific labeling based on current roster status
        remaining = POSITION_LIMITS[pos] - counts[pos]
        if counts[pos] == 0:
            baseline_text = "(need starter)"
        elif remaining == 1:
            baseline_text = "(need 1 more)"
        else:
            baseline_text = f"(need {remaining} more)"
    else:
        # Use worst current starter as baseline
        baseline_text = "(roster complete)"
        worst_starter_slot = counts[pos] - 1  # 0-indexed, so if you have 2 RBs, worst is index 1
        if worst_starter_slot < len(pos_players):
            baseline_player = pos_players[worst_starter_slot]
        else:
            baseline_player = pos_players[0] if len(pos_players) > 0 else None
    
    if not baseline_player:
        return
    
    # Find best available player at current pick
    best_current = None
    best_survival = 0
    
    for i, player in enumerate(pos_players[:15]):  # Check top 15 players
        if i < survival_matrix.shape[0] and pick_number < survival_matrix.shape[1]:
            surv = survival_matrix[i, pick_number]
            if surv > 0.05 and surv > best_survival:  # >5% survival (lower threshold)
                best_survival = surv
                best_current = player
    
    if not best_current:
        # If no one available at current pick, show best player overall
        if len(pos_players) > 0:
            best_current = pos_players[0]
        else:
            return
    
    # Show position name with baseline info
    print(f"{pos}: {baseline_text}")
    
    # Show current pick first, then future picks  
    all_picks = [pick_number] + [p for p in SNAKE_PICKS if p > pick_number][:2]
    
    # Track already shown players to avoid duplicates
    shown_players = set()
    
    # Process each pick (current + future)
    for pick in all_picks:
        if pick >= survival_matrix.shape[1]:
            continue
        
        # Only show players with ≥50% survival probability
        candidates = []
        
        # Check more players for later picks when top players are gone
        if pick <= 30:
            check_limit = min(25, len(pos_players))
        elif pick <= 60:
            check_limit = min(40, len(pos_players))
        else:
            check_limit = min(60, len(pos_players))
        
        for i in range(check_limit):
            if i < survival_matrix.shape[0]:
                player = pos_players[i]
                surv = survival_matrix[i, pick] if pick < survival_matrix.shape[1] else 0.0
                
                # Only include if ≥50% survival
                if surv >= 0.5:
                    # Skip if already shown in earlier pick
                    if player.name in shown_players:
                        continue
                    
                    # Calculate relative value as % of baseline player
                    rel_value = (player.points / baseline_player.points) * 100
                    
                    is_favorite = player.name in favorites
                    candidates.append((player, rel_value, surv, is_favorite))
        
        if not candidates:
            continue
        
        # Sort by relative value and take top 5
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:5]
        
        # Format output for this pick
        pick_outputs = []
        prev_value = None
        
        for player, rel_value, surv, is_fav in top_candidates:
            # Add to shown players set
            shown_players.add(player.name)
            
            # Check for value cliff (15%+ drop from previous - lowered threshold)
            cliff_marker = ""
            if prev_value and (prev_value - rel_value) >= 15:
                cliff_marker = " 🔴"
            
            fav_marker = "⭐" if is_fav else ""
            pick_outputs.append(f"{player.name}{fav_marker} ({rel_value:.0f}%){cliff_marker}")
            prev_value = rel_value
        
        if pick_outputs:
            # Show up to 3 on first line, rest on continuation if needed
            if len(pick_outputs) <= 3:
                print(f"  → Pick {pick:3d}: {', '.join(pick_outputs)}")
            else:
                print(f"  → Pick {pick:3d}: {', '.join(pick_outputs[:3])},")
                print(f"              {', '.join(pick_outputs[3:])}")
    
    print()
    
    # Analytics data capture (if enabled) - using new recording function
    if USE_ENVELOPES:  # Only capture if envelopes enabled (as per original feedback)
        # Record all candidates shown for all pick windows
        for pick in all_picks:
            if pick >= survival_matrix.shape[1]:
                continue
            
            outlook_rows = []
            # Reconstruct which candidates were shown for this pick
            for i in range(min(60, len(pos_players))):
                if i < survival_matrix.shape[0]:
                    player = pos_players[i]
                    surv = survival_matrix[i, pick] if pick < survival_matrix.shape[1] else 0.0
                    
                    # Only record players that would have been shown (≥50% survival)
                    if surv >= 0.5 and player.name not in shown_players:
                        rel_value = (player.points / baseline_player.points) if baseline_player and baseline_player.points > 0 else 1.0
                        outlook_rows.append({
                            "name": player.name,
                            "rel_baseline": rel_value,
                            "survival": surv
                        })
            
            # Record if we have any candidates for this window
            if outlook_rows:
                _record_pos_outlook(pick_number, f"Pick {pick}", pos, outlook_rows[:5])  # Top 5 shown


def show_value_decay_analysis(pick_idx: int, counts: Dict[str, int]):
    """Show fantasy point value drop-off if you wait to draft each position."""
    if pick_idx + 1 >= len(SNAKE_PICKS):
        return  # Not enough future picks to analyze
    
    next_picks = SNAKE_PICKS[pick_idx:pick_idx + 4]  # Current + next 3 (or fewer if near end)
    if len(next_picks) < 2:
        return
    
    print("VALUE DECAY ANALYSIS (fantasy point drop-off if you wait):")
    print()
    
    # For each position, calculate expected best available player at each pick
    position_decays = []
    
    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players = pos_sorted(PLAYERS, pos)
        survival_matrix = SURVIVAL_PROBS.get(pos)
        
        if survival_matrix is None or survival_matrix.size == 0:
            continue
        
        # Calculate expected fantasy points of best available player at each pick
        expected_values = []
        best_players = []
        
        for pick_idx, pick in enumerate(next_picks):
            if pick >= survival_matrix.shape[1]:
                expected_values.append(0)
                best_players.append("None")
                continue
                
            # Find most likely best available player
            best_availability_score = 0
            best_player_name = "None"
            best_player_points = 0
            
            for i, player in enumerate(pos_players[:50]):  # Check enough players to find realistic options
                if i >= survival_matrix.shape[0]:
                    break
                    
                survival_prob = survival_matrix[i, pick]
                
                # Apply 50% survival filter to ALL picks - only show realistic options
                if survival_prob < 0.5:
                    continue
                
                # Calculate probability all better players are taken
                prob_available = survival_prob
                for j in range(i):  # All better players
                    if j < survival_matrix.shape[0]:
                        prob_available *= (1 - survival_matrix[j, pick])
                
                # Use availability score to find best player, but store actual fantasy points
                if prob_available > best_availability_score:
                    best_availability_score = prob_available
                    best_player_name = player.name
                    best_player_points = player.points
            
            expected_values.append(best_player_points)  # Store actual fantasy points, not probability-weighted
            best_players.append(best_player_name)
        
        if len(expected_values) >= 2 and expected_values[0] > 0:
            # Calculate percentage drops
            pct_drops = []
            for i in range(1, len(expected_values)):
                pct_drop = ((expected_values[0] - expected_values[i]) / expected_values[0]) * 100
                pct_drops.append(pct_drop)
            
            total_pct_drop = ((expected_values[0] - expected_values[-1]) / expected_values[0]) * 100
            position_decays.append({
                'pos': pos,
                'best_player': best_players[0],
                'current_points': expected_values[0],
                'pct_drops': pct_drops,
                'total_pct_drop': total_pct_drop,
                'picks': next_picks[1:]  # Skip current pick for display
            })
    
    # Sort by total drop (biggest loss first)
    position_decays.sort(key=lambda x: x['total_pct_drop'], reverse=True)
    
    # Display percentage value drops
    current_pick = next_picks[0]
    
    for p in position_decays:
        drops_str = " → ".join([f"-{drop:.0f}%" for drop in p['pct_drops']])
        picks_str = "→".join(["current"] + [str(pick) for pick in p['picks']])
        
        # Show survival probability for debugging unrealistic availability
        pos_players = pos_sorted(PLAYERS, p['pos'])
        survival_matrix = SURVIVAL_PROBS.get(p['pos'])
        
        # Find the player's survival probability
        player_survival = "N/A"
        for i, player in enumerate(pos_players):
            if player.name == p['best_player'] and i < survival_matrix.shape[0] and current_pick < survival_matrix.shape[1]:
                player_survival = f"{survival_matrix[i, current_pick]:.0%}"
                break
        
        print(f"{p['pos']} (best now: {p['best_player']}, {p['current_points']:.0f}pts, {player_survival} survival): {drops_str} ({picks_str})")
    
    print()
    
    # Analytics data capture (if enabled) - using new recording function
    if USE_ENVELOPES:  # Only capture if envelopes enabled (as per original feedback)
        for p in position_decays:
            if p['pct_drops'] and len(p['pct_drops']) > 0:
                # Record current vs next pick decay
                now_pts = p['current_points']
                next_pts = p['current_points'] * (1 - p['pct_drops'][0]/100) if p['pct_drops'] else now_pts
                _record_value_decay(current_pick, p['pos'], now_pts, next_pts, None)


def show_clean_pick_analysis(pick_idx: int, pick_number: int, counts: Dict[str, int], favorites: set):
    """Show clean pick analysis for fast/stable modes."""
    print(f"\nPICK {pick_number} ANALYSIS (Pick #{pick_idx + 1} of {len(SNAKE_PICKS)})")
    print(f"Current roster: RB={counts['RB']}, WR={counts['WR']}, QB={counts['QB']}, TE={counts['TE']}")
    print("=" * 60)
    
    # Get recommended position
    value, recommended_pos = dp_optimize(
        pick_idx, counts["RB"], counts["WR"], counts["QB"], counts["TE"]
    )
    
    print(f"\nRECOMMENDATION: Draft {recommended_pos.upper()}")
    print()
    
    # Show recommended position first, then others
    all_positions = ["RB", "WR", "QB", "TE"]
    
    # Show recommended position first
    if recommended_pos and counts[recommended_pos] < POSITION_LIMITS[recommended_pos]:
        available_players = get_available_players(recommended_pos, pick_number, favorites)
        show_position_players(recommended_pos, recommended_pos, available_players, pick_number)
    
    # Show value decay analysis
    show_value_decay_analysis(pick_idx, counts)
    
    # Note: "also available" sections removed to reduce redundancy with positional outlook
    # Users can reference the positional outlook section below for other positions
    
    # Positional outlook section
    print("─" * 60)
    print("POSITIONAL OUTLOOK:")
    print("(Percentages below = fantasy points relative to baseline player, NOT survival probability)")
    print()
    
    # Show all positions regardless of roster status
    for pos in ["RB", "WR", "QB", "TE"]:
        show_positional_outlook(pos, counts, pick_number, favorites)
    
    print("─" * 60)
    print("⭐ = Your targets | 🔴 = Value cliff (15%+ drop)")
    print("Shows players with ≥50% survival probability")
    print()
    print("Key points:")
    print("- RECOMMENDED section shows AVAIL% (survival probability)")
    print("- POSITIONAL OUTLOOK shows fantasy point percentages (relative to baseline)")
    print("- Players shown in outlook have ≥50% chance of being available at that pick")
    print("- ⭐ marks your favorite players from the cheat sheet")
    print("- 🔴 indicates a value cliff (15%+ drop from previous player)")
    print()
    
    # Note: Analytics capture moved to final optimal strategy section to ensure accuracy


def clear_capture_examples():
    """Clear captured examples for a fresh run."""
    global CAPTURE_EXAMPLES
    CAPTURE_EXAMPLES = {
        'monte_carlo_pick': None,
        'ladder_ev_calculation': None,
        'dp_decision_comparison': None
    }


def main():
    """Main execution function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description="Fantasy Draft Optimizer - Monte Carlo Simulation"
    )
    
    # Primary modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", 
        action="store_true",
        help="Quick mode: 100 simulations, basic output"
    )
    mode_group.add_argument(
        "--standard", 
        action="store_true",
        help="Standard mode: 5000 simulations, full analytics and exports (default)"
    )
    mode_group.add_argument(
        "--debug", 
        action="store_true",
        help="Debug mode: Standard + debug output and visualizations"
    )
    
    # Draft position
    parser.add_argument(
        "--position",
        type=int,
        default=5,
        choices=range(1, 15),
        help="Your draft position (1-14, default: 5)"
    )
    
    # Optional overrides
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--espn-file",
        type=str,
        default="data/probability-models-draft/espn_projections_20250814.csv",
        help="Path to ESPN projections CSV (default: espn_projections_20250814.csv)"
    )
    parser.add_argument(
        "--envelope-file",
        type=str,
        help="Path to envelope projections CSV for uncertainty analysis"
    )
    
    args = parser.parse_args()

    # Apply command-line overrides
    global RANDOMNESS_LEVEL, USE_ENVELOPES, EXPORT_FORMAT, ENVELOPE_FILE, SNAKE_PICKS
    
    # Calculate snake picks based on position
    SNAKE_PICKS = calculate_snake_picks(args.position, LEAGUE_SIZE, 7)  # 7 rounds for optimization
    print(f"\nDraft position #{args.position} in {LEAGUE_SIZE}-team league")
    print(f"Your picks (first 7 rounds): {SNAKE_PICKS}")
    
    # Set mode configurations
    if args.quick:
        num_sims = 100
        export_csv = False
        visualize = False
        debug_mode = False
        USE_ENVELOPES = False
        print("Mode: QUICK (100 simulations, basic output)")
    elif args.debug:
        num_sims = 5000
        export_csv = True
        visualize = True
        debug_mode = True
        USE_ENVELOPES = True  # Enable everything in debug
        print("Mode: DEBUG (5000 simulations, full analytics + debug output)")
    else:  # Standard mode (default)
        num_sims = 5000
        export_csv = True
        visualize = False
        debug_mode = False
        USE_ENVELOPES = True  # Standard includes all analytics
        print("Mode: STANDARD (5000 simulations, full analytics and exports)")
    
    # Apply envelope file if provided
    if args.envelope_file:
        ENVELOPE_FILE = args.envelope_file
        USE_ENVELOPES = True
        print(f"Using envelope projections: {ENVELOPE_FILE}")
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Clear capture examples and analytics data for fresh run
    clear_capture_examples()
    global _capture_pick_candidates, _capture_value_decay, _capture_pos_outlook
    _capture_pick_candidates.clear()
    _capture_value_decay.clear() 
    _capture_pos_outlook.clear()

    logger.info("Loading player data...")
    
    # Load players using simplified hierarchical matching system
    players = load_and_merge_data(args.espn_file)
    print(f"Loaded {len(players)} players from {args.espn_file}")
    
    # Load favorites from CSV
    favorites = set()
    favorites_file = "data/draft_day_cheat_sheet.csv"
    
    try:
        if os.path.exists(favorites_file):
            df = pd.read_csv(favorites_file)
            if 'player_name' in df.columns:
                favorites = set(df['player_name'].str.strip())
                print(f"Loaded {len(favorites)} favorite players from cheat sheet")
            else:
                logger.warning("cheat sheet missing 'player_name' column")
    except Exception as e:
        logger.info(f"Could not load favorites: {e}")

    # Show top players by position in debug mode
    if debug_mode:
        print("\nTop 3 players by position (with fantasy points):")
        for pos in ["RB", "WR", "QB", "TE"]:
            # CRITICAL FIX: Use points-sorted order to match ladder calculations
            top = pos_sorted(players, pos)[:3]
            names = ", ".join([f"{p.name}({p.points:.0f})" for p in top])
            print(f"  {pos}: {names}")

    # Display current configuration
    print(f"\n{'='*60}")
    print("SIMULATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Data Source: ESPN rankings")
    print(
        f"Randomness Level: {RANDOMNESS_LEVEL} ({'Very Low' if RANDOMNESS_LEVEL <= 0.15 else 'Low' if RANDOMNESS_LEVEL <= 0.25 else 'Moderate' if RANDOMNESS_LEVEL <= 0.4 else 'High' if RANDOMNESS_LEVEL <= 0.6 else 'Very High'})"
    )
    print(f"Candidate Pool Size: {CANDIDATE_POOL_SIZE} players")
    print(f"Number of Simulations: {num_sims}")
    print(f"{'='*60}")

    logger.info(f"Running {num_sims} Monte Carlo simulations...")
    player_survival = monte_carlo_survival_realistic(
        players, num_sims, export_simulation_data=export_csv, data_source="espn", enhanced_stats=USE_ENVELOPES
    )

    # Setup global data for DP optimization
    global PLAYERS, SURVIVAL_PROBS
    PLAYERS = players
    
    # Convert enhanced stats back to simple probabilities for DP optimization
    if USE_ENVELOPES:
        simple_survival_probs = {}
        for key, data_list in player_survival.items():
            simple_survival_probs[key] = np.mean(data_list) if data_list else 0.0
        SURVIVAL_PROBS = get_position_survival_matrix(players, simple_survival_probs)
    else:
        SURVIVAL_PROBS = get_position_survival_matrix(players, player_survival)

    # Handle optional exports and visualization
    if export_csv:
        export_mc_results_to_csv(
            players, player_survival, SNAKE_PICKS, num_sims, "espn", args.seed, USE_ENVELOPES
        )

    if visualize:
        if VISUALIZATION_AVAILABLE:
            create_simple_dashboard(players, player_survival, SNAKE_PICKS, num_sims)
        else:
            print(
                "Visualization not available. Install matplotlib: pip install matplotlib"
            )

    # Show technical summary for stable/debug modes
    if export_csv or debug_mode:
        show_technical_summary()
    
    if debug_mode:
        show_top_players_survival()

    # Stability sweep removed for simplicity
    # if args.stability_sweep:
    #     run_stability_sweep(players)
    #     return  # Exit after stability sweep

    print("\nOptimizing draft strategy...")

    # Build optimal sequence with debug output
    sequence, counts = [], {"RB": 0, "WR": 0, "QB": 0, "TE": 0}

    for pick_idx in range(len(SNAKE_PICKS)):
        # Route output based on mode
        if debug_mode:
            show_pick_analysis(pick_idx, SNAKE_PICKS[pick_idx], counts)
        else:  # fast or stable mode
            show_clean_pick_analysis(pick_idx, SNAKE_PICKS[pick_idx], counts, favorites)

        value, position = dp_optimize(
            pick_idx, counts["RB"], counts["WR"], counts["QB"], counts["TE"]
        )
        sequence.append(position)
        if position:
            counts[position] += 1
            if debug_mode:
                print(f"\n>>> DP DECISION: Draft {position} (Total EV={value:.1f})")

    # Calculate final expected value
    dp_optimize.cache_clear()
    total_value, _ = dp_optimize(0, 0, 0, 0, 0)

    # Display results
    print("\n" + "=" * 50)
    print("OPTIMAL DRAFT STRATEGY SUMMARY")
    print("=" * 50)
    print(f"Expected Total Points: {total_value:.2f}")
    print(f"Monte Carlo Simulations: {num_sims}")
    print(f"Snake Draft Picks: {SNAKE_PICKS}")
    print()

    # Add explanation header
    print("Pick-by-Pick Strategy:")
    print("Format: Pick X: POSITION (likely: Player Name, Survival=X%, Best-Available=X%)")
    print()
    
    # Show pick-by-pick strategy with likely players and availability probabilities
    for i, (pick, position) in enumerate(zip(SNAKE_PICKS, sequence)):
        if position:
            # CRITICAL FIX: Use points-sorted order to match ladder calculations
            pos_players = pos_sorted(players, position)

            # Find most likely available player with availability probability
            best_availability, likely_player, likely_survival = 0, None, 0.0
            for j, player in enumerate(pos_players):
                survival_prob = player_survival.get((player.unique_id, pick), 0.0)
                # Calculate probability all better players (by points) are taken
                taken_prob = 1.0
                for h in range(j):  # All players ranked higher by points
                    better_player = pos_players[h]
                    taken_prob *= 1 - player_survival.get(
                        (better_player.unique_id, pick), 0.0
                    )

                availability = survival_prob * taken_prob
                if availability > best_availability:
                    best_availability, likely_player, likely_survival = availability, player, survival_prob

            if likely_player and likely_survival > 0.05:
                player_info = f" (likely: {likely_player.name}, {likely_survival:.0%} available)"
            else:
                player_info = " (no clear favorite)"

            print(f"Pick {pick:2d}: {position}{player_info}")
            
            # Analytics capture for FINAL optimal strategy (not intermediate analysis)
            if USE_ENVELOPES and position and likely_player:
                # Get top candidates for this optimal position
                top_candidates = []
                for j, player in enumerate(pos_players[:5]):  # Top 5 candidates
                    survival_prob = player_survival.get((player.unique_id, pick), 0.0)
                    top_candidates.append({
                        "name": player.name,
                        "proj_pts": player.points,
                        "avail": survival_prob
                    })
                # Record the optimal position's candidates
                _record_candidates(pick, position, top_candidates)
                
        else:
            print(f"Pick {pick:2d}: ")

    print("\nPosition Summary:")
    for pos in ["RB", "WR", "QB", "TE"]:
        print(f"  {pos}: {sequence.count(pos)}/{POSITION_LIMITS[pos]}")
    
    print("\n" + "─" * 60)
    print("PERCENTAGE MEANINGS:")
    print("• Survival %: Chance the player is still available when your pick arrives")
    print("• Positional outlook %: Fantasy points relative to baseline player")
    print("• Best-Available: Chance this player will be your best option among available players")
    print("─" * 60)
    
    # Export analytics data if enabled
    _export_pick_run()


if __name__ == "__main__":
    main()
