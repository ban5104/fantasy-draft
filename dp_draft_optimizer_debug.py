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

# Configuration
SNAKE_PICKS = [5, 24, 33, 52, 61, 80, 89]  # 14-team league picks
MC_SIMS = 10  # Monte Carlo simulations (increase for production)
POSITION_LIMITS = {'RB': 3, 'WR': 2, 'QB': 1, 'TE': 1}
DEBUG = True  # Toggle detailed output

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

def monte_carlo_survival_realistic(players: List[Player], num_sims: int = MC_SIMS) -> Dict[Tuple[str, int], float]:
    """
    More realistic Monte Carlo - respects ESPN rankings with noise.
    Returns survival_probs[(player_name, pick_number)] = probability
    """
    survival_counts = {}
    
    for sim in range(num_sims):
        available = players.copy()
        
        for pick_num in range(1, 90):  # First 90 picks
            if pick_num in SNAKE_PICKS:
                # Record who's available at our picks
                for player in available:
                    key = (player.name, pick_num)
                    if key not in survival_counts:
                        survival_counts[key] = 0
                    survival_counts[key] += 1
            
            # Simulate other team picking
            if available and pick_num not in SNAKE_PICKS:
                # Pick based on ESPN rank with some randomness
                # Top 3 candidates with weighted probability
                n_candidates = min(3, len(available))
                candidates = available[:n_candidates]
                
                # Weight by rank (1st gets highest weight)
                weights = [1.0 / (i + 1) for i in range(len(candidates))]
                weights = np.array(weights) / sum(weights)
                
                # Pick one
                choice_idx = np.random.choice(len(candidates), p=weights)
                available.pop(choice_idx)
    
    # Convert to probabilities
    survival_probs = {}
    for key, count in survival_counts.items():
        survival_probs[key] = count / num_sims
    
    return survival_probs

def get_position_survival_matrix(players: List[Player], survival_probs: Dict) -> Dict[str, np.ndarray]:
    """Convert player-level survival to position matrices for ladder EV."""
    position_players = {}
    position_matrices = {}
    
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pos_players = [p for p in players if p.position == pos]
        position_players[pos] = pos_players
        
        # Build survival matrix for this position
        max_pick = max(SNAKE_PICKS) + 10
        matrix = np.zeros((len(pos_players), max_pick + 1))
        
        for i, player in enumerate(pos_players):
            for pick in SNAKE_PICKS:
                matrix[i, pick] = survival_probs.get((player.name, pick), 0.0)
        
        position_matrices[pos] = matrix
    
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
        
        if contribution > 0.1 and j < slot + 4:  # Show meaningful contributions
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
            for line in debug_info[:5]:  # Show top 5 lines
                print(line)
            
            # Calculate delta to next pick
            if pick_idx + 1 < len(SNAKE_PICKS):
                next_pick = SNAKE_PICKS[pick_idx + 1]
                next_ev, _ = ladder_ev_debug(pos, next_pick, slot, PLAYERS, SURVIVAL_PROBS)
                delta = ev - next_ev
                print(f"  Delta (now vs pick {next_pick}): {delta:.1f}")
    
    # Show decision
    if position_evs:
        best_pos = max(position_evs, key=position_evs.get)
        print(f"\n>>> DECISION: Draft {best_pos} (EV={position_evs[best_pos]:.1f})")
    
    return position_evs

@lru_cache(maxsize=None)
def dp_optimize(pick_idx: int, rb_count: int, wr_count: int, qb_count: int, te_count: int) -> Tuple[float, str]:
    """DP recurrence: F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}"""
    
    if pick_idx >= len(SNAKE_PICKS):
        return 0.0, ""
    
    global PLAYERS, SURVIVAL_PROBS
    
    current_pick = SNAKE_PICKS[pick_idx]
    counts = {'RB': rb_count, 'WR': wr_count, 'QB': qb_count, 'TE': te_count}
    
    best_value = 0.0
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
            surv_str = ""
            for pick in SNAKE_PICKS:
                prob = survival_matrix[i, pick] if pick < survival_matrix.shape[1] else 0
                surv_str += f"P{pick}:{prob:.0%} "
            print(f"  {player.name[:20]:20} ({player.points:.0f}pts) - {surv_str}")

def main():
    """Main execution function."""
    print("Loading player data...")
    players = load_and_merge_data()
    print(f"Loaded {len(players)} players")
    
    # Show top players by position
    if DEBUG:
        print("\nTop 3 players by position (with fantasy points):")
        for pos in ['RB', 'WR', 'QB', 'TE']:
            top = [p for p in players if p.position == pos][:3]
            names = ", ".join([f"{p.name}({p.points:.0f})" for p in top])
            print(f"  {pos}: {names}")
    
    print(f"\nRunning {MC_SIMS} Monte Carlo simulations...")
    player_survival = monte_carlo_survival_realistic(players, MC_SIMS)
    
    # Convert to position matrices
    global SURVIVAL_PROBS
    SURVIVAL_PROBS = get_position_survival_matrix(players, player_survival)
    
    if DEBUG:
        show_top_players_survival()
    
    print("\nOptimizing draft strategy...")
    
    # Set global data
    global PLAYERS
    PLAYERS = players
    
    # Build optimal sequence with debug output
    sequence = []
    counts = {'RB': 0, 'WR': 0, 'QB': 0, 'TE': 0}
    
    for pick_idx in range(len(SNAKE_PICKS)):
        if DEBUG:
            position_evs = show_pick_analysis(pick_idx, SNAKE_PICKS[pick_idx], counts)
        
        # Get optimal position from DP
        value, position = dp_optimize(
            pick_idx, counts['RB'], counts['WR'], counts['QB'], counts['TE']
        )
        
        sequence.append(position)
        if position:
            counts[position] += 1
    
    # Clear cache and solve for final value
    dp_optimize.cache_clear()
    total_value, _ = dp_optimize(0, 0, 0, 0, 0)
    
    print("\n" + "="*50)
    print("OPTIMAL DRAFT STRATEGY SUMMARY")
    print("="*50)
    print(f"Expected Total Points: {total_value:.2f}")
    print(f"Snake Draft Picks: {SNAKE_PICKS}")
    print()
    
    for i, (pick, position) in enumerate(zip(SNAKE_PICKS, sequence)):
        pos_count = sequence[:i+1].count(position)
        # Show what player we'd likely get
        pos_players = [p for p in players if p.position == position]
        likely_player = pos_players[pos_count - 1] if pos_count <= len(pos_players) else None
        player_name = f" (likely: {likely_player.name})" if likely_player else ""
        print(f"Pick {pick:2d}: {position}{player_name}")
    
    print("\nPosition Summary:")
    for pos in ['RB', 'WR', 'QB', 'TE']:
        count = sequence.count(pos)
        print(f"  {pos}: {count}/{POSITION_LIMITS[pos]}")

if __name__ == "__main__":
    main()