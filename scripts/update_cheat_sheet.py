#!/usr/bin/env python3
"""
Fantasy Draft Cheat Sheet Updater

Automatically updates the draft day cheat sheet CSV with the latest data from
probability models and rankings. This script merges data from ESPN projections,
ADP data, and other sources to keep the cheat sheet current.
"""

import pandas as pd
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheatSheetUpdater:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.probability_dir = self.data_dir / "probability-models-draft"
        
    def normalize_player_name(self, name):
        """Normalize player names for matching across different data sources."""
        if pd.isna(name):
            return ""
        
        # Remove common suffixes and normalize
        name = str(name).strip()
        name = re.sub(r'\s+(Jr\.|Sr\.|III|IV|V)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name)
        
        # Handle common name variations
        replacements = {
            'Ken Walker': 'Kenneth Walker',
            'D.J.': 'DJ',
            'A.J.': 'AJ',
            'T.J.': 'TJ',
            'J.K.': 'JK',
        }
        
        for old, new in replacements.items():
            if name.startswith(old):
                name = name.replace(old, new, 1)
                
        return name
        
    def create_name_mapping(self, full_names_df, abbreviated_names_df):
        """Create a mapping between abbreviated and full names."""
        mapping = {}
        
        # Common first name mappings
        first_name_map = {
            'J.': ['Ja\'Marr', 'Justin', 'Josh', 'Jahmyr', 'James', 'Joe', 'Jaylen', 'Jayden', 'Jordan', 'Jalen', 'Jonathan', 'Jaxon', 'Jerome', 'Jake', 'Javonte', 'Jauan', 'Jonnu'],
            'B.': ['Bijan', 'Breece', 'Brock', 'Brian', 'Brandon', 'Baker', 'Blake'],
            'S.': ['Saquon', 'Sam', 'Stefon', 'Sean'],
            'C.': ['Christian', 'CeeDee', 'Calvin', 'Cooper', 'Chase', 'Caleb', 'Chris', 'Cam', 'Cameron', 'Courtland', 'Chuba', 'Cole'],
            'D.': ['Derrick', 'Davante', 'DeVonta', 'Drake', 'DK', 'David', 'Dak', 'Dalton', 'Darnell', 'DJ', 'Diontae', 'Dallas'],
            'T.': ['Tyreek', 'Trey', 'Terry', 'Travis', 'Tony', 'Tyler', 'Tank', 'Tua', 'TJ', 'Tucker', 'Trevor', 'Tyjae'],
            'A.': ['Amon-Ra', 'Aaron', 'Alvin', 'Austin', 'Amari', 'Adam', 'Ashton', 'AJ'],
            'K.': ['Kyren', 'Kenneth', 'Kyle', 'Kyler', 'Khalil', 'Keenan', 'Keon'],
            'M.': ['Malik', 'Mike', 'Michael', 'Marvin', 'Matthew'],
            'R.': ['Rashee', 'Rachaad', 'Rhamondre', 'Rome', 'Rashid', 'Rashod', 'Romeo', 'Ricky', 'Ray'],
            'G.': ['George', 'Garrett', 'Geno'],
            'L.': ['Lamar', 'Ladd', 'Luther'],
            'N.': ['Nico', 'Nick', 'Najee'],
            'P.': ['Patrick', 'Puka', 'Pat'],
            'Z.': ['Zach', 'Zay']
        }
        
        for _, abb_row in abbreviated_names_df.iterrows():
            abb_name = str(abb_row.get('player', abb_row.get('name', '')))
            if not abb_name or pd.isna(abb_name):
                continue
                
            # Extract parts of abbreviated name
            parts = abb_name.split()
            if len(parts) < 2:
                continue
                
            first_initial = parts[0]
            last_name = ' '.join(parts[1:])
            
            # Try to find matching full name
            for _, full_row in full_names_df.iterrows():
                full_name = str(full_row.get('player', full_row.get('player_name', '')))
                if not full_name or pd.isna(full_name):
                    continue
                    
                # Check if last name matches
                if last_name.lower() in full_name.lower():
                    # Check if first initial matches or is in mapping
                    if first_initial in first_name_map:
                        full_first = full_name.split()[0] if full_name.split() else ''
                        if full_first in first_name_map[first_initial]:
                            mapping[abb_name] = full_name
                            break
                    elif full_name.lower().startswith(first_initial.lower().replace('.', '')):
                        mapping[abb_name] = full_name
                        break
        
        return mapping
        
    def find_latest_file(self, pattern):
        """Find the most recent file matching a pattern."""
        files = list(self.probability_dir.glob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return None
        
        # Sort by date in filename or modification time
        latest = max(files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using latest file: {latest.name}")
        return latest
        
    def load_espn_algorithm_data(self):
        """Load ESPN algorithm rankings data."""
        espn_file = self.find_latest_file("espn_algorithm_*.csv")
        if not espn_file:
            logger.warning("No ESPN algorithm data files found!")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(espn_file)
            logger.info(f"Loaded {len(df)} ESPN algorithm rankings from {espn_file.name}")
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            if 'player_name' in df.columns:
                df['player'] = df['player_name']
            elif 'name' in df.columns:
                df['player'] = df['name']
                
            df['player_normalized'] = df['player'].apply(self.normalize_player_name)
            return df[['player', 'player_normalized', 'position', 'overall_rank', 'team']].copy()
            
        except Exception as e:
            logger.error(f"Error loading ESPN algorithm data: {e}")
            return pd.DataFrame()
            
    def load_espn_projections_data(self):
        """Load ESPN projections data."""
        espn_file = self.find_latest_file("espn_projections_*.csv")
        if not espn_file:
            logger.warning("No ESPN projections data files found!")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(espn_file)
            logger.info(f"Loaded {len(df)} ESPN projections from {espn_file.name}")
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            if 'player_name' in df.columns:
                df['player'] = df['player_name']
            elif 'name' in df.columns:
                df['player'] = df['name']
                
            df['player_normalized'] = df['player'].apply(self.normalize_player_name)
            return df[['player', 'player_normalized', 'position', 'overall_rank', 'team']].copy()
            
        except Exception as e:
            logger.error(f"Error loading ESPN projections data: {e}")
            return pd.DataFrame()
    
    def load_adp_data(self, espn_df=None):
        """Load ADP (Average Draft Position) data."""
        adp_file = self.find_latest_file("realtime_adp_*.csv")
        if not adp_file:
            logger.warning("No ADP data files found!")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(adp_file)
            logger.info(f"Loaded {len(df)} ADP rankings from {adp_file.name}")
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Handle different column name variations
            if 'name' in df.columns:
                df['player'] = df['name']
            elif 'player_name' in df.columns:
                df['player'] = df['player_name']
                
            # Clean team field (remove bye week info)
            if 'team' in df.columns:
                df['team'] = df['team'].str.replace(r'\s*\([^)]*\)', '', regex=True)
                
            if 'real_time_adp' in df.columns:
                df['sleeper_adp'] = df['real_time_adp']
            elif 'adp' in df.columns:
                df['sleeper_adp'] = df['adp']
            
            # Create name mapping if ESPN data is available
            if espn_df is not None and not espn_df.empty:
                name_mapping = self.create_name_mapping(espn_df, df)
                logger.info(f"Created name mapping for {len(name_mapping)} players")
                
                # Apply mapping to convert abbreviated names to full names
                df['player_mapped'] = df['player'].map(name_mapping).fillna(df['player'])
                df['player'] = df['player_mapped']
                df = df.drop('player_mapped', axis=1)
                
            df['player_normalized'] = df['player'].apply(self.normalize_player_name)
            return df[['player', 'player_normalized', 'sleeper_adp']].copy()
            
        except Exception as e:
            logger.error(f"Error loading ADP data: {e}")
            return pd.DataFrame()
            
    def load_actual_draft_data(self):
        """Load actual draft results data."""
        draft_file = self.find_latest_file("actual_draft_results_*.csv")
        if not draft_file:
            logger.warning("No actual draft results files found!")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(draft_file)
            logger.info(f"Loaded {len(df)} actual draft rankings from {draft_file.name}")
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            
            if 'player_name' in df.columns:
                df['player'] = df['player_name']
            elif 'name' in df.columns:
                df['player'] = df['name']
                
            df['player_normalized'] = df['player'].apply(self.normalize_player_name)
            return df[['player', 'player_normalized', 'overall_rank']].copy()
            
        except Exception as e:
            logger.error(f"Error loading actual draft data: {e}")
            return pd.DataFrame()
            
    def load_current_cheat_sheet(self):
        """Load the current cheat sheet to preserve manual additions."""
        cheat_sheet_path = self.data_dir / "draft_day_cheat_sheet.csv"
        
        if not cheat_sheet_path.exists():
            logger.info("No existing cheat sheet found, will create new one")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(cheat_sheet_path)
            logger.info(f"Loaded existing cheat sheet with {len(df)} players")
            df['player_normalized'] = df['PLAYER'].apply(self.normalize_player_name)
            return df
        except Exception as e:
            logger.error(f"Error loading current cheat sheet: {e}")
            return pd.DataFrame()
    
    def get_team_bye_weeks(self):
        """Return a mapping of team abbreviations to bye weeks."""
        # 2025 NFL bye weeks (you'll need to update this annually)
        return {
            'ATL': 5, 'BUF': 12, 'CAR': 11, 'CHI': 5, 'CIN': 10, 'CLE': 10,
            'DAL': 10, 'DEN': 7, 'DET': 8, 'GB': 10, 'HOU': 7, 'IND': 14,
            'JAX': 11, 'KC': 10, 'LV': 8, 'LAC': 8, 'LAR': 6, 'MIA': 5,
            'MIN': 6, 'NE': 11, 'NO': 12, 'NYG': 11, 'NYJ': 5, 'PHI': 9,
            'PIT': 8, 'SF': 14, 'SEA': 5, 'TB': 5, 'TEN': 5, 'WAS': 12
        }
    
    def merge_data_sources(self, espn_alg_df, espn_proj_df, adp_df, draft_df, current_df):
        """Merge all data sources to create updated cheat sheet."""
        logger.info("Merging data sources...")
        
        # Start with current cheat sheet as base (preserves manual additions)
        if not current_df.empty:
            result_df = current_df.copy()
            logger.info(f"Starting with {len(result_df)} existing players")
        else:
            result_df = pd.DataFrame()
        
        # Get bye week mapping
        bye_weeks = self.get_team_bye_weeks()
        
        # Merge ESPN Algorithm data
        if not espn_alg_df.empty:
            for _, espn_row in espn_alg_df.iterrows():
                player_norm = espn_row['player_normalized']
                
                # Find matching player in current cheat sheet
                existing_match = None
                if not result_df.empty:
                    matches = result_df[result_df['player_normalized'] == player_norm]
                    if not matches.empty:
                        existing_match = matches.index[0]
                
                if existing_match is not None:
                    # Update existing player
                    result_df.loc[existing_match, 'ESPN_ALG'] = espn_row['overall_rank']
                    if 'team' in espn_row and pd.notna(espn_row['team']):
                        result_df.loc[existing_match, 'TEAM'] = espn_row['team']
                        result_df.loc[existing_match, 'BYE'] = bye_weeks.get(espn_row['team'], '')
                else:
                    # Add new player
                    new_player = {
                        'PLAYER': espn_row['player'],
                        'POS': espn_row['position'],
                        'TEAM': espn_row.get('team', ''),
                        'ESPN_ALG': espn_row['overall_rank'],
                        'ESPN_PROJ': '',
                        'ADP_SLEEPER': '',
                        'ACTUAL_DRAFT_20250823': '',
                        'BYE': bye_weeks.get(espn_row.get('team', ''), ''),
                        'player_normalized': player_norm
                    }
                    result_df = pd.concat([result_df, pd.DataFrame([new_player])], ignore_index=True)
        
        # Merge ESPN Projections data
        if not espn_proj_df.empty:
            for _, espn_row in espn_proj_df.iterrows():
                player_norm = espn_row['player_normalized']
                
                # Find matching player
                if not result_df.empty:
                    matches = result_df[result_df['player_normalized'] == player_norm]
                    if not matches.empty:
                        idx = matches.index[0]
                        result_df.loc[idx, 'ESPN_PROJ'] = espn_row['overall_rank']
        
        # Merge ADP data (Sleeper)
        if not adp_df.empty:
            for _, adp_row in adp_df.iterrows():
                player_norm = adp_row['player_normalized']
                
                # Find matching player
                if not result_df.empty:
                    matches = result_df[result_df['player_normalized'] == player_norm]
                    if not matches.empty:
                        idx = matches.index[0]
                        result_df.loc[idx, 'ADP_SLEEPER'] = round(float(adp_row['sleeper_adp']), 1)
        
        # Merge actual draft results (RTSports equivalent)
        if not draft_df.empty:
            for _, draft_row in draft_df.iterrows():
                player_norm = draft_row['player_normalized']
                
                # Find matching player
                if not result_df.empty:
                    matches = result_df[result_df['player_normalized'] == player_norm]
                    if not matches.empty:
                        idx = matches.index[0]
                        result_df.loc[idx, 'ACTUAL_DRAFT_20250823'] = draft_row['overall_rank']
        
        # Remove the normalized column before saving
        if 'player_normalized' in result_df.columns:
            result_df = result_df.drop('player_normalized', axis=1)
        
        # Sort by ESPN Algorithm ranking
        if 'ESPN_ALG' in result_df.columns:
            result_df = result_df.sort_values('ESPN_ALG', na_position='last')
        
        logger.info(f"Final cheat sheet contains {len(result_df)} players")
        return result_df
    
    def update_cheat_sheet(self, output_file=None):
        """Main method to update the cheat sheet."""
        logger.info("Starting cheat sheet update...")
        
        # Load ESPN data first (needed for name mapping)
        espn_alg_df = self.load_espn_algorithm_data()
        espn_proj_df = self.load_espn_projections_data()
        
        # Combine ESPN dataframes for name mapping
        espn_combined = pd.concat([espn_alg_df, espn_proj_df], ignore_index=True).drop_duplicates(subset=['player'])
        
        # Load ADP data with name mapping
        adp_df = self.load_adp_data(espn_combined)
        
        # Load other data sources
        draft_df = self.load_actual_draft_data()
        current_df = self.load_current_cheat_sheet()
        
        # Merge data
        updated_df = self.merge_data_sources(espn_alg_df, espn_proj_df, adp_df, draft_df, current_df)
        
        if updated_df.empty:
            logger.error("No data to update cheat sheet!")
            return False
        
        # Save updated cheat sheet
        if output_file is None:
            output_file = self.data_dir / "draft_day_cheat_sheet.csv"
        
        try:
            updated_df.to_csv(output_file, index=False)
            logger.info(f"Updated cheat sheet saved to {output_file}")
            
            # Print summary
            print(f"\n✅ Cheat Sheet Updated Successfully!")
            print(f"📊 Total players: {len(updated_df)}")
            print(f"📁 Saved to: {output_file}")
            print(f"🗓️  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving cheat sheet: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Update fantasy football draft cheat sheet")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        updater = CheatSheetUpdater(args.data_dir)
        success = updater.update_cheat_sheet(args.output)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Update cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()