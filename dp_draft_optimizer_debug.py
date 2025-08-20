#!/usr/bin/env python3
"""
DP Draft Optimizer with DEBUG MODE - shows detailed calculations.
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from functools import lru_cache
from typing import Dict, List, Tuple, NamedTuple
import itertools
import argparse
try:
    from visualize_mc import create_simple_dashboard
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Configuration
SNAKE_PICKS = [5, 24, 33, 52, 61, 80, 89]  # 14-team league picks
POSITION_LIMITS = {'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1}

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
    espn = pd.read_csv('data/espn_projections_20250814.csv')
    points = pd.read_csv('data/rankings_top300_20250814.csv')
    
    # Create lookup for fantasy points
    points_lookup = {row['PLAYER']: row['FANTASY_PTS'] for _, row in points.iterrows()}
    
    players = []
    for _, row in espn.iterrows():
        # Fuzzy match player names
        best_match = None
        best_score = 0
        for points_name in points_lookup.keys():
            score = fuzz.ratio(row['player_name'], points_name)
            if score > best_score:
                best_score = score
                best_match = points_name
        
        # Use matched points or default
        fantasy_points = points_lookup.get(best_match, 0.0) if best_score > 70 else 0.0
        
        players.append(Player(
            name=row['player_name'],
            position=row['position'],
            points=fantasy_points,
            overall_rank=row['overall_rank']
        ))
    
    return sorted(players, key=lambda p: p.overall_rank)

def export_mc_results_to_csv(players, survival_probs, snake_picks, num_sims):
    """Export Monte Carlo results to CSV files for Jupyter analysis."""
    import pandas as pd
    from datetime import datetime
    
    # 1. Player survival probabilities
    player_data = []
    for player in players:
        row = {
            'player_name': player.name,
            'position': player.position,
            'fantasy_points': player.points,
            'overall_rank': player.overall_rank
        }
        # Add survival probability for each pick
        for pick in snake_picks:
            prob = survival_probs.get((player.name, pick), 0.0)
            row[f'survival_pick_{pick}'] = prob
        player_data.append(row)
    
    df_players = pd.DataFrame(player_data)
    df_players.to_csv('mc_player_survivals.csv', index=False)
    
    # 2. Position summary data
    position_data = []
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pos_players = [p for p in players if p.position == pos]
        for pick in snake_picks:
            pick_survivals = []
            for player in pos_players[:10]:  # Top 10
                prob = survival_probs.get((player.name, pick), 0.0)
                pick_survivals.append(prob)
            
            position_data.append({
                'position': pos,
                'pick': pick,
                'avg_survival_top10': np.mean(pick_survivals) if pick_survivals else 0,
                'max_survival_top10': np.max(pick_survivals) if pick_survivals else 0,
                'min_survival_top10': np.min(pick_survivals) if pick_survivals else 0
            })
    
    df_positions = pd.DataFrame(position_data)
    df_positions.to_csv('mc_position_summary.csv', index=False)
    
    # 3. Configuration metadata
    config_data = {
        'num_simulations': [num_sims],
        'snake_picks': [str(snake_picks)],
        'position_limits': [str(POSITION_LIMITS)],
        'export_timestamp': [datetime.now().isoformat()],
        'total_players': [len(players)]
    }
    df_config = pd.DataFrame(config_data)
    df_config.to_csv('mc_config.csv', index=False)
    
    print(f"\nExported Monte Carlo results to CSV files:")
    print(f"  - mc_player_survivals.csv ({len(player_data)} players)")
    print(f"  - mc_position_summary.csv ({len(position_data)} position/pick combinations)")
    print(f"  - mc_config.csv (simulation metadata)")
    print(f"\nLoad in Jupyter with: pd.read_csv('mc_player_survivals.csv')")

def monte_carlo_survival_realistic(players: List[Player], num_sims: int) -> Dict[Tuple[str, int], float]:
    """
    More realistic Monte Carlo with position scarcity modeling.
    Returns survival_probs[(player_name, pick_number)] = probability
    """
    survival_counts = {}
    
    # Initialize counts for all players at all our picks
    for player in players:
        for pick in SNAKE_PICKS:
            survival_counts[(player.name, pick)] = 0
    
    for sim in range(num_sims):
        available = players.copy()
        teams_roster_counts = [{pos: 0 for pos in ['QB', 'RB', 'WR', 'TE']} for _ in range(14)]
        
        for pick_num in range(1, 90):  # First 90 picks
            if not available:
                break
                
            if pick_num in SNAKE_PICKS:
                # Record who's available at our picks
                for player in available:
                    key = (player.name, pick_num)
                    survival_counts[key] += 1
            else:
                # Simulate other team picking with position scarcity
                round_num = (pick_num - 1) // 14
                if round_num % 2 == 0:  # Odd rounds (1, 3, 5...)
                    team_idx = (pick_num - 1) % 14
                else:  # Even rounds (2, 4, 6...)
                    team_idx = 13 - ((pick_num - 1) % 14)
                team_roster = teams_roster_counts[team_idx]
                
                # Filter candidates by position needs
                position_weights = {
                    'QB': 3.0 if team_roster['QB'] == 0 else 0.5,
                    'RB': 2.0 if team_roster['RB'] < 2 else 1.0,
                    'WR': 2.0 if team_roster['WR'] < 2 else 1.0,
                    'TE': 2.5 if team_roster['TE'] == 0 else 0.5
                }
                
                # Weight candidates by both rank and position need
                candidate_scores = []
                for i, player in enumerate(available[:15]):  # Consider top 15
                    rank_weight = 1.0 / (i + 1)
                    pos_weight = position_weights.get(player.position, 1.0)
                    # Add some randomness
                    noise = np.random.normal(1.0, 0.3)
                    total_score = rank_weight * pos_weight * max(0.1, noise)
                    candidate_scores.append((player, total_score))
                
                if candidate_scores:
                    # Pick based on weighted scores
                    candidates, scores = zip(*candidate_scores)
                    scores = np.array(scores)
                    weights = scores / scores.sum()
                    
                    choice_idx = np.random.choice(len(candidates), p=weights)
                    picked_player = candidates[choice_idx]
                    available.remove(picked_player)
                    
                    # Update team roster
                    team_roster[picked_player.position] += 1
    
    # Convert to probabilities
    survival_probs = {}
    for key, count in survival_counts.items():
        survival_probs[key] = count / num_sims
    
    return survival_probs

def get_position_survival_matrix(players: List[Player], survival_probs: Dict[Tuple[str, int], float]) -> Dict[str, np.ndarray]:
    """Convert player-level survival to position matrices for ladder EV."""
    position_matrices = {}
    
    for pos in ['RB', 'WR', 'QB', 'TE']:
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
            print(f"Position {pos}: {len(pos_players)} players, avg survival: {avg_survival:.3f}")
    
    return position_matrices

def ladder_ev_debug(position: str, pick_number: int, slot: int, players: List[Player], 
                    survival_probs: Dict[str, np.ndarray]) -> Tuple[float, List[str]]:
    """Compute expected value with debug info."""
    pos_players = [p for p in players if p.position == position]
    if not pos_players:
        return 0.0, []
    
    survival = survival_probs[position]
    debug_info = []
    expected_value = 0.0
    
    # Start from slot-th best available player
    debug_info.append(f"  Position {position}, Slot {slot} at Pick {pick_number}:")
    
    for j in range(slot - 1, min(len(pos_players), 10)):  # Show top 10 only for debug
        if j >= survival.shape[0]:
            break
            
        player = pos_players[j]
        player_value = player.points
        
        # Probability this player is available
        surv_prob = survival[j, pick_number] if pick_number < survival.shape[1] else 0.0
        
        # Probability all better players are taken
        taken_prob = 1.0
        for h in range(slot - 1, j):
            if h < survival.shape[0] and pick_number < survival.shape[1]:
                taken_prob *= (1 - survival[h, pick_number])
        
        contribution = player_value * surv_prob * taken_prob
        expected_value += contribution
        
        if contribution > 0.01 or j < slot + 2:  # Show top players + meaningful contributions
            debug_info.append(f"    {player.name}: {player_value:.1f}pts × {surv_prob:.2f}surv × {taken_prob:.2f}gone = {contribution:.1f}")
    
    debug_info.append(f"    Total EV: {expected_value:.1f}")
    return expected_value, debug_info

def show_pick_analysis(pick_idx: int, pick_number: int, counts: Dict[str, int]):
    """Show detailed analysis for a specific pick."""
    global PLAYERS, SURVIVAL_PROBS
    
    print(f"\n{'='*60}")
    print(f"PICK {pick_number} ANALYSIS (Pick #{pick_idx + 1} of {len(SNAKE_PICKS)})")
    print(f"Current roster: RB={counts['RB']}, WR={counts['WR']}, QB={counts['QB']}, TE={counts['TE']}")
    print(f"{'='*60}")
    
    # Calculate EV for each position
    position_evs = {}
    for pos in ['RB', 'WR', 'QB', 'TE']:
        if counts[pos] < POSITION_LIMITS[pos]:
            slot = counts[pos] + 1
            ev, debug_info = ladder_ev_debug(pos, pick_number, slot, PLAYERS, SURVIVAL_PROBS)
            position_evs[pos] = ev
            
            print(f"\n{pos} Analysis:")
            for line in debug_info:  # Show all debug lines
                print(line)
            
            # Calculate delta to next pick
            if pick_idx + 1 < len(SNAKE_PICKS):
                next_pick = SNAKE_PICKS[pick_idx + 1]
                next_ev, _ = ladder_ev_debug(pos, next_pick, slot, PLAYERS, SURVIVAL_PROBS)
                delta = ev - next_ev
                print(f"  Delta (now vs pick {next_pick}): {delta:.1f}")
    
    return position_evs

@lru_cache(maxsize=None)
def dp_optimize(pick_idx: int, rb_count: int, wr_count: int, qb_count: int, te_count: int) -> Tuple[float, str]:
    """DP recurrence: F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}"""
    
    if pick_idx >= len(SNAKE_PICKS):
        return 0.0, ""
    
    global PLAYERS, SURVIVAL_PROBS
    
    current_pick = SNAKE_PICKS[pick_idx]
    counts = {'RB': rb_count, 'WR': wr_count, 'QB': qb_count, 'TE': te_count}
    
    best_value = -float('inf')  # Start with negative infinity
    best_position = ""
    
    for pos in ['RB', 'WR', 'QB', 'TE']:
        if counts[pos] < POSITION_LIMITS[pos]:
            slot = counts[pos] + 1
            ev, _ = ladder_ev_debug(pos, current_pick, slot, PLAYERS, SURVIVAL_PROBS)
            
            new_counts = counts.copy()
            new_counts[pos] += 1
            
            future_value, _ = dp_optimize(
                pick_idx + 1,
                new_counts['RB'],
                new_counts['WR'], 
                new_counts['QB'],
                new_counts['TE']
            )
            
            total_value = ev + future_value
            
            if total_value > best_value:
                best_value = total_value
                best_position = pos
    
    # If no valid position found, return 0
    if best_value == -float('inf'):
        return 0.0, ""
    
    return best_value, best_position

def show_top_players_survival():
    """Show survival probabilities for top players at each position."""
    global PLAYERS, SURVIVAL_PROBS
    
    print("\n" + "="*60)
    print("TOP PLAYER SURVIVAL PROBABILITIES")
    print("="*60)
    
    for pos in ['RB', 'WR', 'QB', 'TE']:
        print(f"\n{pos} Top 5:")
        pos_players = [p for p in PLAYERS if p.position == pos][:5]
        survival_matrix = SURVIVAL_PROBS[pos]
        
        for i, player in enumerate(pos_players):
            if i >= survival_matrix.shape[0]:
                break
            surv_str = ""
            for pick in SNAKE_PICKS:
                prob = survival_matrix[i, pick] if pick < survival_matrix.shape[1] else 0
                surv_str += f"P{pick}:{prob:.0%} "
            print(f"  {player.name[:20]:20} ({player.points:.0f}pts) - {surv_str}")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DP Draft Optimizer')
    parser.add_argument('--sims', type=int, default=1000, 
                       help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('--debug', action='store_true', default=True,
                       help='Enable debug mode (default: True)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate Monte Carlo visualization dashboard')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots as PNG files')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export Monte Carlo results to CSV files')
    args = parser.parse_args()
    
    num_sims = args.sims
    debug_mode = args.debug
    
    print("Loading player data...")
    players = load_and_merge_data()
    print(f"Loaded {len(players)} players")
    
    # Show top players by position
    if debug_mode:
        print("\nTop 3 players by position (with fantasy points):")
        for pos in ['RB', 'WR', 'QB', 'TE']:
            top = [p for p in players if p.position == pos][:3]
            names = ", ".join([f"{p.name}({p.points:.0f})" for p in top])
            print(f"  {pos}: {names}")
    
    print(f"\nRunning {num_sims} Monte Carlo simulations...")
    player_survival = monte_carlo_survival_realistic(players, num_sims)
    
    # Convert to position matrices
    global SURVIVAL_PROBS
    SURVIVAL_PROBS = get_position_survival_matrix(players, player_survival)
    
    # Export to CSV if requested
    if args.export_csv:
        export_mc_results_to_csv(players, player_survival, SNAKE_PICKS, num_sims)
    
    # Generate visualizations if requested
    if args.visualize and VISUALIZATION_AVAILABLE:
        create_simple_dashboard(players, player_survival, SNAKE_PICKS, num_sims)
    elif args.visualize and not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib: pip install matplotlib")
    
    if debug_mode:
        show_top_players_survival()
    
    print("\nOptimizing draft strategy...")
    
    # Set global data
    global PLAYERS
    PLAYERS = players
    
    # Build optimal sequence with debug output
    sequence = []
    counts = {'RB': 0, 'WR': 0, 'QB': 0, 'TE': 0}
    
    for pick_idx in range(len(SNAKE_PICKS)):
        if debug_mode:
            show_pick_analysis(pick_idx, SNAKE_PICKS[pick_idx], counts)
        
        # Get optimal position from DP
        value, position = dp_optimize(
            pick_idx, counts['RB'], counts['WR'], counts['QB'], counts['TE']
        )
        
        sequence.append(position)
        if position:
            counts[position] += 1
            
        # Show the actual decision from DP
        if debug_mode and position:
            print(f"\n>>> DP DECISION: Draft {position} (Total EV={value:.1f})")
    
    # Clear cache and solve for final value
    dp_optimize.cache_clear()
    total_value, _ = dp_optimize(0, 0, 0, 0, 0)
    
    print("\n" + "="*50)
    print("OPTIMAL DRAFT STRATEGY SUMMARY")
    print("="*50)
    print(f"Expected Total Points: {total_value:.2f}")
    print(f"Monte Carlo Simulations: {num_sims}")
    print(f"Snake Draft Picks: {SNAKE_PICKS}")
    print()
    
    for i, (pick, position) in enumerate(zip(SNAKE_PICKS, sequence)):
        if position:  # Only show valid positions
            pos_count = sequence[:i+1].count(position)
            # Show what player we'd likely get based on survival probabilities
            pos_players = [p for p in players if p.position == position]
            
            # Find most likely available player based on survival probabilities
            likely_player = None
            best_expected_availability = 0
            
            for player in pos_players:
                survival_prob = player_survival.get((player.name, pick), 0.0)
                # Calculate "taken probability" for higher-ranked players
                taken_prob = 1.0
                for better_player in pos_players:
                    if better_player.overall_rank < player.overall_rank:
                        better_survival = player_survival.get((better_player.name, pick), 0.0)
                        taken_prob *= (1 - better_survival)
                
                expected_availability = survival_prob * taken_prob
                if expected_availability > best_expected_availability:
                    best_expected_availability = expected_availability
                    likely_player = player
            
            if likely_player and best_expected_availability > 0.01:  # Only show if >1% chance
                player_name = f" (likely: {likely_player.name})"
            else:
                player_name = " (no clear favorite)"
            
            print(f"Pick {pick:2d}: {position}{player_name}")
        else:
            print(f"Pick {pick:2d}: ")
    
    print("\nPosition Summary:")
    for pos in ['RB', 'WR', 'QB', 'TE']:
        count = sequence.count(pos)
        print(f"  {pos}: {count}/{POSITION_LIMITS[pos]}")

if __name__ == "__main__":
    main()