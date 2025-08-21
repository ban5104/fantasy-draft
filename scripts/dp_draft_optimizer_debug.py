#!/usr/bin/env python3
"""
DP Draft Optimizer with DEBUG MODE - shows detailed calculations.
"""

import argparse
import itertools
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

# ==============================================================================
# CONFIGURATION - Adjust these parameters to control the simulation
# ==============================================================================

# Your draft picks in the snake draft (14-team league example)
SNAKE_PICKS = [5, 24, 33, 52, 61, 80, 89]  # Update with your actual picks

# Maximum roster slots you need to fill
POSITION_LIMITS = {"RB": 3, "WR": 2, "QB": 1, "TE": 1}

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


class Player(NamedTuple):
    name: str
    position: str
    points: float
    overall_rank: int


def load_and_merge_data() -> List[Player]:
    """Load ESPN projections and fantasy points, merge via fuzzy matching."""
    # Validate required CSV files exist before attempting to load
    espn_file = "data/espn_projections_20250814.csv"
    points_file = "data/rankings_top300_20250814.csv"

    if not os.path.exists(espn_file):
        print(f"ERROR: Required ESPN projections file not found: {espn_file}")
        print(
            "Please ensure the data directory contains the ESPN projections CSV file."
        )
        sys.exit(1)

    if not os.path.exists(points_file):
        print(f"ERROR: Required rankings file not found: {points_file}")
        print("Please ensure the data directory contains the rankings CSV file.")
        sys.exit(1)

    try:
        espn = pd.read_csv(espn_file)
        points = pd.read_csv(points_file)
    except Exception as e:
        print(f"ERROR: Failed to load CSV files: {e}")
        sys.exit(1)

    # Create lookup for fantasy points
    points_lookup = {row["PLAYER"]: row["FANTASY_PTS"] for _, row in points.iterrows()}

    players = []
    low_quality_matches = []  # Track poor fuzzy matches for logging

    for _, row in espn.iterrows():
        # Fuzzy match player names
        best_match = None
        best_score = 0
        for points_name in points_lookup.keys():
            score = fuzz.ratio(row["player_name"], points_name)
            if score > best_score:
                best_score = score
                best_match = points_name

        # Use matched points or default
        fantasy_points = points_lookup.get(best_match, 0.0) if best_score > 70 else 0.0

        # Log poor quality matches for data quality review
        if best_score < 70 and best_match:
            low_quality_matches.append(
                {
                    "espn_name": row["player_name"],
                    "best_match": best_match,
                    "match_score": best_score,
                }
            )

        players.append(
            Player(
                name=row["player_name"],
                position=row["position"],
                points=fantasy_points,
                overall_rank=row["overall_rank"],
            )
        )

    # Log fuzzy matching quality issues for manual review
    if low_quality_matches:
        print(
            f"\nWARNING: {len(low_quality_matches)} players had poor fuzzy matching (< 70% similarity):"
        )
        for match in low_quality_matches[:10]:  # Show first 10 to avoid spam
            print(
                f"  ESPN: '{match['espn_name']}' -> Best: '{match['best_match']}' ({match['match_score']}%)"
            )
        if len(low_quality_matches) > 10:
            print(
                f"  ... and {len(low_quality_matches) - 10} more. Consider manual data review."
            )

    return sorted(players, key=lambda p: p.overall_rank)


def export_mc_results_to_csv(players, survival_probs, snake_picks, num_sims):
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
        row.update(
            {
                f"survival_pick_{pick}": survival_probs.get((player.name, pick), 0.0)
                for pick in snake_picks
            }
        )
        return row

    # 1. Player survival data
    player_data = [create_player_row(p) for p in players]
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(player_data).to_csv(
        os.path.join(output_dir, "mc_player_survivals.csv"), index=False
    )

    # 2. Position summary data
    position_data = []
    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players = [p for p in players if p.position == pos][:10]
        for pick in snake_picks:
            survivals = [survival_probs.get((p.name, pick), 0.0) for p in pos_players]
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
        os.path.join(output_dir, "mc_position_summary.csv"), index=False
    )

    # 3. Configuration metadata
    config_data = {
        "num_simulations": [num_sims],
        "snake_picks": [str(snake_picks)],
        "position_limits": [str(POSITION_LIMITS)],
        "export_timestamp": [datetime.now().isoformat()],
        "total_players": [len(players)],
    }
    pd.DataFrame(config_data).to_csv(
        os.path.join(output_dir, "mc_config.csv"), index=False
    )

    print(f"\nExported Monte Carlo results to CSV files:")
    print(
        f"  - {os.path.join(output_dir, 'mc_player_survivals.csv')} ({len(player_data)} players)"
    )
    print(
        f"  - {os.path.join(output_dir, 'mc_position_summary.csv')} ({len(position_data)} position/pick combinations)"
    )
    print(f"  - {os.path.join(output_dir, 'mc_config.csv')} (simulation metadata)")
    print(
        f"\nLoad in Jupyter with: pd.read_csv('{os.path.join(output_dir, 'mc_player_survivals.csv')}')"
    )


def monte_carlo_survival_realistic(
    players: List[Player], num_sims: int, export_simulation_data: bool = False
) -> Dict[Tuple[str, int], float]:
    """
    More realistic Monte Carlo with position scarcity modeling.
    Returns survival_probs[(player_name, pick_number)] = probability
    """
    survival_counts = {(p.name, pick): 0 for p in players for pick in SNAKE_PICKS}
    simulation_picks = []

    for sim in range(num_sims):
        available = players.copy()

        for pick_num in range(1, 90):
            if not available:
                break

            if pick_num in SNAKE_PICKS:
                for player in available:
                    survival_counts[(player.name, pick_num)] += 1
            else:
                # Dynamic candidate pool sizing based on pick timing
                if pick_num <= 30:
                    pool_size = CANDIDATE_POOL_SIZE
                elif pick_num <= 60:
                    pool_size = min(len(available), int(CANDIDATE_POOL_SIZE * 1.5))
                else:
                    pool_size = min(len(available), int(CANDIDATE_POOL_SIZE * 2))
                candidates = available[:pool_size]
                scores = [
                    (1.0 / (i + 1)) * max(0.1, np.random.normal(1.0, RANDOMNESS_LEVEL))
                    for i, p in enumerate(candidates)
                ]

                if scores:
                    weights = np.array(scores) / sum(scores)
                    choice_idx = np.random.choice(len(candidates), p=weights)
                    picked_player = candidates[choice_idx]
                    available.remove(picked_player)

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
            os.path.join(output_dir, "mc_simulation_picks.csv"), index=False
        )
        print(
            f"Exported {len(simulation_picks)} individual simulation picks to {os.path.join(output_dir, 'mc_simulation_picks.csv')}"
        )

    return {key: count / num_sims for key, count in survival_counts.items()}


def get_position_survival_matrix(
    players: List[Player], survival_probs: Dict[Tuple[str, int], float]
) -> Dict[str, np.ndarray]:
    """Convert player-level survival to position matrices for ladder EV."""
    position_matrices = {}

    for pos in ["RB", "WR", "QB", "TE"]:
        pos_players = [p for p in players if p.position == pos]

        # Build survival matrix for this position
        max_pick = max(SNAKE_PICKS) + 10
        matrix = np.zeros((len(pos_players), max_pick + 1))

        for i, player in enumerate(pos_players):
            for pick in SNAKE_PICKS:
                matrix[i, pick] = survival_probs.get((player.name, pick), 0.0)

        position_matrices[pos] = matrix

        # Data validation - warn if all survival probabilities are 0
        if matrix.sum() == 0:
            print(f"WARNING: No survival probabilities found for position {pos}")
        else:
            avg_survival = matrix[matrix > 0].mean() if matrix.sum() > 0 else 0
            print(
                f"Position {pos}: {len(pos_players)} players, avg survival: {avg_survival:.3f}"
            )

    return position_matrices


def ladder_ev_debug(
    position: str,
    pick_number: int,
    slot: int,
    players: List[Player],
    survival_probs: Dict[str, np.ndarray],
) -> Tuple[float, List[str]]:
    """Compute expected value with debug info."""
    pos_players = [p for p in players if p.position == position]
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
    return expected_value, debug_info


def show_pick_analysis(pick_idx: int, pick_number: int, counts: Dict[str, int]):
    """Show detailed analysis for a specific pick."""
    global PLAYERS, SURVIVAL_PROBS

    print(f"\n{'='*60}")
    print(f"PICK {pick_number} ANALYSIS (Pick #{pick_idx + 1} of {len(SNAKE_PICKS)})")
    print(
        f"Current roster: RB={counts['RB']}, WR={counts['WR']}, QB={counts['QB']}, TE={counts['TE']}"
    )
    print(f"{'='*60}")

    # Calculate EV for each position
    position_evs = {}
    for pos in ["RB", "WR", "QB", "TE"]:
        if counts[pos] < POSITION_LIMITS[pos]:
            slot = counts[pos] + 1
            ev, debug_info = ladder_ev_debug(
                pos, pick_number, slot, PLAYERS, SURVIVAL_PROBS
            )
            position_evs[pos] = ev

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

    return position_evs


@lru_cache(maxsize=None)
def dp_optimize(
    pick_idx: int, rb_count: int, wr_count: int, qb_count: int, te_count: int
) -> Tuple[float, str]:
    """DP recurrence: F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}"""

    if pick_idx >= len(SNAKE_PICKS):
        return 0.0, ""

    global PLAYERS, SURVIVAL_PROBS

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

                if total_value > best_value:
                    best_value = total_value
                    best_position = pos

            except Exception as e:
                print(
                    f"ERROR: Exception in DP optimization for {pos} at pick {current_pick}: {e}"
                )
                continue

    return (best_value, best_position) if best_value != -float("inf") else (0.0, "")


def show_top_players_survival():
    """Show survival probabilities for top players at each position."""
    global PLAYERS, SURVIVAL_PROBS

    print("\n" + "=" * 60)
    print("TOP PLAYER SURVIVAL PROBABILITIES")
    print("=" * 60)

    if not PLAYERS or not SURVIVAL_PROBS:
        print("WARNING: No players or survival data available")
        return

    for pos in ["RB", "WR", "QB", "TE"]:
        print(f"\n{pos} Top 5:")
        pos_players = [p for p in PLAYERS if p.position == pos][:5]

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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="DP Draft Optimizer with Monte Carlo Simulation"
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode (default: True)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate Monte Carlo visualization dashboard",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots as PNG files",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export Monte Carlo results to CSV files",
    )
    parser.add_argument(
        "--export-simulations",
        action="store_true",
        help="Export individual simulation pick data for scatter plots",
    )

    # Simulation tuning parameters (override config file values)
    parser.add_argument(
        "--randomness", type=float, help="Randomness level 0.0-1.0 (override config)"
    )
    parser.add_argument(
        "--pool-size", type=int, help="Candidate pool size (override config)"
    )
    args = parser.parse_args()

    # Apply command-line overrides
    global RANDOMNESS_LEVEL, CANDIDATE_POOL_SIZE
    if args.randomness is not None:
        RANDOMNESS_LEVEL = max(0.0, min(1.0, args.randomness))  # Clamp to valid range
    if args.pool_size is not None:
        CANDIDATE_POOL_SIZE = max(3, min(50, args.pool_size))  # Reasonable bounds

    print("Loading player data...")
    players = load_and_merge_data()
    print(f"Loaded {len(players)} players")

    # Show top players by position in debug mode
    if args.debug:
        print("\nTop 3 players by position (with fantasy points):")
        for pos in ["RB", "WR", "QB", "TE"]:
            top = [p for p in players if p.position == pos][:3]
            names = ", ".join([f"{p.name}({p.points:.0f})" for p in top])
            print(f"  {pos}: {names}")

    # Display current configuration
    print(f"\n{'='*60}")
    print("SIMULATION CONFIGURATION")
    print(f"{'='*60}")
    print(
        f"Randomness Level: {RANDOMNESS_LEVEL} ({'Very Low' if RANDOMNESS_LEVEL <= 0.15 else 'Low' if RANDOMNESS_LEVEL <= 0.25 else 'Moderate' if RANDOMNESS_LEVEL <= 0.4 else 'High' if RANDOMNESS_LEVEL <= 0.6 else 'Very High'})"
    )
    print(f"Candidate Pool Size: {CANDIDATE_POOL_SIZE} players")
    print(f"Number of Simulations: {args.sims}")
    print(f"{'='*60}")

    print(f"\nRunning {args.sims} Monte Carlo simulations...")
    player_survival = monte_carlo_survival_realistic(
        players, args.sims, export_simulation_data=args.export_simulations
    )

    # Setup global data for DP optimization
    global PLAYERS, SURVIVAL_PROBS
    PLAYERS = players
    SURVIVAL_PROBS = get_position_survival_matrix(players, player_survival)

    # Handle optional exports and visualization
    if args.export_csv:
        export_mc_results_to_csv(players, player_survival, SNAKE_PICKS, args.sims)

    if args.visualize:
        if VISUALIZATION_AVAILABLE:
            create_simple_dashboard(players, player_survival, SNAKE_PICKS, args.sims)
        else:
            print(
                "Visualization not available. Install matplotlib: pip install matplotlib"
            )

    if args.debug:
        show_top_players_survival()

    print("\nOptimizing draft strategy...")

    # Build optimal sequence with debug output
    sequence, counts = [], {"RB": 0, "WR": 0, "QB": 0, "TE": 0}

    for pick_idx in range(len(SNAKE_PICKS)):
        if args.debug:
            show_pick_analysis(pick_idx, SNAKE_PICKS[pick_idx], counts)

        value, position = dp_optimize(
            pick_idx, counts["RB"], counts["WR"], counts["QB"], counts["TE"]
        )
        sequence.append(position)
        if position:
            counts[position] += 1
            if args.debug:
                print(f"\n>>> DP DECISION: Draft {position} (Total EV={value:.1f})")

    # Calculate final expected value
    dp_optimize.cache_clear()
    total_value, _ = dp_optimize(0, 0, 0, 0, 0)

    # Display results
    print("\n" + "=" * 50)
    print("OPTIMAL DRAFT STRATEGY SUMMARY")
    print("=" * 50)
    print(f"Expected Total Points: {total_value:.2f}")
    print(f"Monte Carlo Simulations: {args.sims}")
    print(f"Snake Draft Picks: {SNAKE_PICKS}")
    print()

    # Show pick-by-pick strategy with likely players
    for i, (pick, position) in enumerate(zip(SNAKE_PICKS, sequence)):
        if position:
            pos_players = [p for p in players if p.position == position]

            # Find most likely available player
            best_availability, likely_player = 0, None
            for player in pos_players:
                survival_prob = player_survival.get((player.name, pick), 0.0)
                taken_prob = 1.0
                for better in pos_players:
                    if better.overall_rank < player.overall_rank:
                        taken_prob *= 1 - player_survival.get((better.name, pick), 0.0)

                availability = survival_prob * taken_prob
                if availability > best_availability:
                    best_availability, likely_player = availability, player

            player_info = (
                f" (likely: {likely_player.name})"
                if likely_player and best_availability > 0.01
                else " (no clear favorite)"
            )
            print(f"Pick {pick:2d}: {position}{player_info}")
        else:
            print(f"Pick {pick:2d}: ")

    print("\nPosition Summary:")
    for pos in ["RB", "WR", "QB", "TE"]:
        print(f"  {pos}: {sequence.count(pos)}/{POSITION_LIMITS[pos]}")


if __name__ == "__main__":
    main()
