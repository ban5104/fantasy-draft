#!/usr/bin/env python3

# Simple test to verify favorites loading logic
import pandas as pd
import os

def test_favorites_loading():
    """Test the favorites loading functionality."""
    favorites = set()
    try:
        favorites_file = "data/draft_day_cheat_sheet.csv"
        if os.path.exists(favorites_file):
            favorites_df = pd.read_csv(favorites_file)
            if 'player_name' in favorites_df.columns:
                favorites = set(favorites_df['player_name'].str.strip())
                print(f"Loaded {len(favorites)} favorite players from cheat sheet")
                print("Sample favorites:", list(favorites)[:5])
            else:
                print("WARNING: draft_day_cheat_sheet.csv found but missing 'player_name' column")
        else:
            print("No cheat sheet found")
    except Exception as e:
        print(f"Note: Could not load favorites from cheat sheet: {e}")
    
    return favorites

if __name__ == "__main__":
    favorites = test_favorites_loading()
    print(f"Total favorites loaded: {len(favorites)}")