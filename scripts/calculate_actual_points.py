#!/usr/bin/env python3
"""
Calculate actual projected fantasy points for different draft strategies.
This script reads the CSV data and calculates real totals, not DP expected values.
"""

import pandas as pd
import sys

def load_rankings():
    """Load player rankings from CSV."""
    try:
        df = pd.read_csv('data/rankings_top300_20250814.csv')
        print(f"Loaded {len(df)} players from rankings CSV")
        return df
    except FileNotFoundError:
        print("ERROR: Could not find rankings CSV file")
        sys.exit(1)

def find_player_points(df, player_name):
    """Find a player's projected points, with fuzzy matching."""
    # Try exact match first
    exact_match = df[df['PLAYER'].str.contains(player_name, case=False, na=False)]
    
    if len(exact_match) > 0:
        player = exact_match.iloc[0]
        return player['FANTASY_PTS'], player['PLAYER']
    
    # Try partial match
    partial_match = df[df['PLAYER'].str.contains(player_name.split()[0], case=False, na=False)]
    
    if len(partial_match) > 0:
        player = partial_match.iloc[0]
        return player['FANTASY_PTS'], player['PLAYER']
    
    return None, None

def calculate_strategy_points(df, strategy_name, players):
    """Calculate total points for a draft strategy."""
    print(f"\n{strategy_name}:")
    print("=" * 50)
    
    total_points = 0
    found_players = []
    
    for i, player_name in enumerate(players, 1):
        points, full_name = find_player_points(df, player_name)
        
        if points is not None:
            total_points += points
            found_players.append((i, full_name, points))
            print(f"Pick {i}: {full_name} - {points:.2f} pts")
        else:
            print(f"Pick {i}: {player_name} - NOT FOUND")
    
    print(f"\nTOTAL PROJECTED POINTS: {total_points:.2f}")
    return total_points, found_players

def main():
    # Load player data
    df = load_rankings()
    
    # Strategy 1: Earlier simulation (RB-RB-RB-QB-TE-WR-WR)
    earlier_strategy = [
        "Bijan Robinson",
        "Jonathan Taylor", 
        "Chase Brown",
        "Patrick Mahomes",
        "Sam LaPorta",
        "Tetairoa McMillan",
        "Chris Godwin"
    ]
    
    # Strategy 2: Later simulation (RB-RB-WR-QB-WR-RB-TE)
    later_strategy = [
        "Bijan Robinson",
        "Jonathan Taylor",
        "Tee Higgins", 
        "Patrick Mahomes",
        "Xavier Worthy",
        "David Montgomery",
        "Mark Andrews"
    ]
    
    # Calculate points for both strategies
    earlier_total, earlier_players = calculate_strategy_points(
        df, "EARLIER STRATEGY (RB-RB-RB-QB-TE-WR-WR)", earlier_strategy
    )
    
    later_total, later_players = calculate_strategy_points(
        df, "LATER STRATEGY (RB-RB-WR-QB-WR-RB-TE)", later_strategy
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Earlier Strategy Total: {earlier_total:.2f} points")
    print(f"Later Strategy Total:   {later_total:.2f} points")
    print(f"Difference:             {later_total - earlier_total:+.2f} points")
    
    if later_total > earlier_total:
        print(f"✅ Later strategy is better by {later_total - earlier_total:.2f} points")
    else:
        print(f"❌ Earlier strategy is better by {earlier_total - later_total:.2f} points")

if __name__ == "__main__":
    main()