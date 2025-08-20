#!/usr/bin/env python3
"""
DP Draft Optimizer using Monte Carlo survival probabilities and ladder EVs.
Implements backward induction over draft states to find optimal position sequence.
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

def monte_carlo_survival(players: List[Player], num_sims: int = MC_SIMS) -> Dict[str, np.ndarray]:
    """Compute survival probabilities for each player at each pick via Monte Carlo."""
    position_players = {}
    for pos in ['RB', 'WR', 'QB', 'TE']:
        position_players[pos] = [p for p in players if p.position == pos]
    
    max_pick = max(SNAKE_PICKS) + 10  # Buffer for draft simulation
    survival_probs = {}
    
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pos_players = position_players[pos]
        probs = np.zeros((len(pos_players), max_pick + 1))
        
        for sim in range(num_sims):
            # Simulate draft order for this position
            draft_order = np.random.permutation(len(pos_players))
            
            for pick in range(1, max_pick + 1):
                # Players drafted this pick (simple model: 1-2 per pick)
                drafted_this_pick = np.random.randint(1, 3)
                
                for i, player_idx in enumerate(draft_order):
                    if i >= pick * drafted_this_pick:
                        # Player survives to this pick
                        probs[player_idx, pick] += 1
        
        # Normalize by number of simulations
        probs = probs / num_sims
        survival_probs[pos] = probs
    
    return survival_probs

def ladder_ev(position: str, pick_number: int, slot: int, players: List[Player], 
              survival_probs: Dict[str, np.ndarray]) -> float:
    """Compute expected value for drafting a position at given pick/slot."""
    pos_players = [p for p in players if p.position == position]
    if not pos_players:
        return 0.0
    
    pick_idx = pick_number - 1  # Convert to 0-indexed
    survival = survival_probs[position]
    
    expected_value = 0.0
    
    # Start from slot-th best available player (0-indexed)
    for j in range(slot - 1, len(pos_players)):
        if j >= survival.shape[0]:
            break
            
        player_value = pos_players[j].points
        
        # Probability this player is available
        surv_prob = survival[j, pick_idx] if pick_idx < survival.shape[1] else 0.0
        
        # Probability all better players are taken
        taken_prob = 1.0
        for h in range(slot - 1, j):  # Players ranked slot-1 to j-1
            if h < survival.shape[0] and pick_idx < survival.shape[1]:
                taken_prob *= (1 - survival[h, pick_idx])
        
        expected_value += player_value * surv_prob * taken_prob
    
    return expected_value

@lru_cache(maxsize=None)
def dp_optimize(pick_idx: int, rb_count: int, wr_count: int, qb_count: int, te_count: int) -> Tuple[float, str]:
    """DP recurrence: F(k,r,w,q,t) = max{ladder_ev + F(k+1,...)}"""
    
    if pick_idx >= len(SNAKE_PICKS):
        return 0.0, ""
    
    # Use global data
    global PLAYERS, SURVIVAL_PROBS
    
    current_pick = SNAKE_PICKS[pick_idx]
    counts = {'RB': rb_count, 'WR': wr_count, 'QB': qb_count, 'TE': te_count}
    
    best_value = 0.0
    best_position = ""
    
    # Try each position we can still draft
    for pos in ['RB', 'WR', 'QB', 'TE']:
        if counts[pos] < POSITION_LIMITS[pos]:
            # Calculate slot (how many of this position we've already drafted + 1)
            slot = counts[pos] + 1
            
            # Expected value from drafting this position
            ev = ladder_ev(pos, current_pick, slot, PLAYERS, SURVIVAL_PROBS)
            
            # Recursive call for remaining picks
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

def solve_draft(players: List[Player], survival_probs: Dict[str, np.ndarray]) -> Tuple[List[str], float]:
    """Solve the draft optimization problem and return optimal sequence."""
    
    # Set global data for DP optimization
    global PLAYERS, SURVIVAL_PROBS
    PLAYERS = players
    SURVIVAL_PROBS = survival_probs
    
    # Clear cache before starting
    dp_optimize.cache_clear()
    
    # Build optimal sequence
    sequence = []
    total_value = 0.0
    counts = {'RB': 0, 'WR': 0, 'QB': 0, 'TE': 0}
    
    for pick_idx in range(len(SNAKE_PICKS)):
        value, position = dp_optimize(
            pick_idx, counts['RB'], counts['WR'], counts['QB'], counts['TE']
        )
        
        if pick_idx == 0:
            total_value = value
            
        sequence.append(position)
        if position:
            counts[position] += 1
    
    return sequence, total_value

def main():
    """Main execution function."""
    print("Loading player data...")
    players = load_and_merge_data()
    print(f"Loaded {len(players)} players")
    
    print(f"\nRunning {MC_SIMS} Monte Carlo simulations...")
    survival_probs = monte_carlo_survival(players, MC_SIMS)
    
    print("Optimizing draft strategy...")
    sequence, expected_value = solve_draft(players, survival_probs)
    
    print("\n" + "="*50)
    print("OPTIMAL DRAFT STRATEGY")
    print("="*50)
    print(f"Expected Total Points: {expected_value:.2f}")
    print(f"Snake Draft Picks: {SNAKE_PICKS}")
    print()
    
    for i, (pick, position) in enumerate(zip(SNAKE_PICKS, sequence)):
        pos_count = sequence[:i+1].count(position)
        print(f"Pick {pick:2d}: {position} (your {pos_count}{['st','nd','rd'][pos_count-1] if pos_count <= 3 else 'th'} {position})")
    
    print("\nPosition Summary:")
    for pos in ['RB', 'WR', 'QB', 'TE']:
        count = sequence.count(pos)
        print(f"  {pos}: {count}/{POSITION_LIMITS[pos]}")

if __name__ == "__main__":
    main()