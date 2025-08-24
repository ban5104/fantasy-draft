#!/usr/bin/env python3
"""
Convert abbreviated player names in ADP file to full names to match other data sources.
"""

import pandas as pd
import re

# Read ESPN files to build comprehensive name mapping
espn_alg = pd.read_csv('data/probability-models-draft/espn_algorithm_20250824.csv')
espn_proj = pd.read_csv('data/probability-models-draft/espn_projections_20250814.csv')
actual_draft = pd.read_csv('data/probability-models-draft/actual_draft_results_20250823.csv')

# Combine all full names
all_players = pd.concat([
    espn_alg[['player_name', 'position', 'team']],
    espn_proj[['player_name', 'position', 'team']],
    actual_draft[['player_name', 'position', 'team']]
]).drop_duplicates(subset=['player_name'])

# Read ADP file
adp = pd.read_csv('data/probability-models-draft/realtime_adp_20250822.csv')

# Clean team field in ADP (remove bye week info)
adp['Team_Clean'] = adp['Team'].str.replace(r'\s*\([^)]*\)', '', regex=True)

def find_full_name(abb_name, abb_team, all_players_df):
    """Find the full name for an abbreviated name."""
    # Handle special cases first
    special_cases = {
        'K. Walker III': 'Kenneth Walker III',
        'B. Robinson Jr.': 'Brian Robinson Jr.',
        'M. Harrison Jr.': 'Marvin Harrison Jr.',
        'B. Thomas Jr.': 'Brian Thomas Jr.',
        'A. St. Brown': 'Amon-Ra St. Brown',
        'P. Mahomes II': 'Patrick Mahomes',
        'K. Pitts Sr.': 'Kyle Pitts Sr.',
        'A. Jones Sr.': 'Aaron Jones Sr.',
        'J. Smith-Njigba': 'Jaxon Smith-Njigba',
        'T. Henderson': 'TreVeyon Henderson',
        'S. LaPorta': 'Sam LaPorta',
        'R. Stevenson': 'Rhamondre Stevenson',
        'D. Smith Sr.': 'Deebo Samuel Sr.',
        'K. Williams': 'Kyren Williams',
        'O. Hampton': 'Omarion Hampton',
        'L. McConkey': 'Ladd McConkey',
        'R. Ridley': 'Calvin Ridley',
        'T. Hockenson': 'T.J. Hockenson',
        'D. Swift': 'D\'Andre Swift',
        'Z. Flowers': 'Zay Flowers',
        'J. Williams': 'Javonte Williams',  # DAL
        'D. Johnson': 'Diontae Johnson',
        'C. Kirk': 'Christian Kirk',
        'C. Watson': 'Christian Watson',
        'B. Aiyuk': 'Brandon Aiyuk',
        'R. Shaheed': 'Rashid Shaheed',
        'M. Pittman Jr.': 'Michael Pittman Jr.',
        'K. Coleman': 'Keon Coleman',
        'J. Jennings': 'Jauan Jennings',
        'K. Shakir': 'Khalil Shakir',
        'J. Downs': 'Josh Downs',
        'R. Bateman': 'Rashod Bateman',
        'X. Legette': 'Xavier Legette',
        'C. Tillman': 'Cedric Tillman',
        'K. Allen': 'Keenan Allen',
        'R. Doubs': 'Romeo Doubs',
        'T. Kraft': 'Tucker Kraft',
        'D. Goedert': 'Dallas Goedert',
        'D. Kincaid': 'Dalton Kincaid',
        'T. Benson': 'Trey Benson',
        'J. Warren': 'Jaylen Warren',
        'J. Mixon': 'Joe Mixon',
        'A. Ekeler': 'Austin Ekeler',
        'J. Dobbins': 'J.K. Dobbins',
        'Q. Judkins': 'Quinshon Judkins',
        'T. Spears': 'Tyjae Spears',
        'T. Etienne Jr.': 'Travis Etienne Jr.',
        'R. White': 'Rachaad White',
        'C. Stroud': 'C.J. Stroud',
        'J. Love': 'Jordan Love',
        'J. Goff': 'Jared Goff',
        'J. Herbert': 'Justin Herbert',
        'C. Williams': 'Caleb Williams',
        'D. Prescott': 'Dak Prescott',
        'T. Tagovailoa': 'Tua Tagovailoa',
        'M. Stafford': 'Matthew Stafford',
        'B. Purdy': 'Brock Purdy',
        'K. Murray': 'Kyler Murray',
        'J. Fields': 'Justin Fields',
        'N. Harris': 'Najee Harris',
        'Z. Charbonnet': 'Zach Charbonnet',
        'T. Allgeier': 'Tyler Allgeier',
        'J. Ford': 'Jerome Ford',
        'J. Wright': 'Jaylen Wright',
        'B. Allen': 'Braelon Allen',
        'N. Chubb': 'Nick Chubb',
        'T. Bigsby': 'Tank Bigsby',
        'J. Mason': 'Jordan Mason',
        'H. Henry': 'Hunter Henry',
        'J. Ferguson': 'Jake Ferguson',
        'M. Andrews': 'Mark Andrews',
        'T. Kelce': 'Travis Kelce',
        'D. Njoku': 'David Njoku',
        'E. Engram': 'Evan Engram',
        'C. Otton': 'Cade Otton',
        'Z. Ertz': 'Zach Ertz',
        'J. Smith': 'Jonnu Smith',
        'P. Freiermuth': 'Pat Freiermuth',
        'I. Likely': 'Isaiah Likely',
        'D. Schultz': 'Dalton Schultz',
        'M. Gesicki': 'Mike Gesicki',
        'T. Lockett': 'Tyler Lockett',
        'G. Smith': 'Geno Smith',
        'S. Darnold': 'Sam Darnold',
        'D. Maye': 'Drake Maye',
        'J. McCarthy': 'J.J. McCarthy',
        'T. Lawrence': 'Trevor Lawrence',
        'B. Young': 'Bryce Young',
        'M. Penix Jr.': 'Michael Penix Jr.',
        'C. Ward': 'Cameron Ward',
        'A. Richardson': 'Anthony Richardson',
        'C. Dicker': 'Cameron Dicker',
        'B. Aubrey': 'Brandon Aubrey',
        'J. Sanders': 'Jason Sanders',
        'T. Bass': 'Tyler Bass',
        'J. Elliott': 'Jake Elliott',
        'C. Boswell': 'Chris Boswell',
        'H. Butker': 'Harrison Butker',
        'C. Santos': 'Cairo Santos',
        'J. Bates': 'Jake Bates',
        'C. McLaughlin': 'Chase McLaughlin',
        'T. Loop': 'Tyler Loop',
        'M. Gay': 'Matt Gay',
        'K. Fairbairn': 'Ka\'imi Fairbairn',
        'J. Karty': 'Joshua Karty',
    }
    
    if abb_name in special_cases:
        return special_cases[abb_name]
    
    # Extract first initial and last name
    parts = abb_name.split()
    if len(parts) < 2:
        return abb_name  # Can't process, return as-is
    
    first_initial = parts[0].replace('.', '').upper()
    last_name = ' '.join(parts[1:])
    
    # Look for matches
    for _, player in all_players_df.iterrows():
        full_name = player['player_name']
        if pd.isna(full_name):
            continue
            
        # Check if last name matches
        if last_name.lower() in full_name.lower():
            # Check if first initial matches
            if full_name.upper().startswith(first_initial):
                # Verify team if possible
                if pd.notna(player['team']) and pd.notna(abb_team):
                    if player['team'] == abb_team:
                        return full_name
                else:
                    return full_name
    
    # Manual mappings for common first initials
    first_name_map = {
        'J.': {
            'Chase': 'Ja\'Marr Chase',
            'Gibbs': 'Jahmyr Gibbs',
            'Jefferson': 'Justin Jefferson',
            'Allen': 'Josh Allen',
            'Jacobs': 'Josh Jacobs',
            'Taylor': 'Jonathan Taylor',
            'Cook': 'James Cook',
            'Burrow': 'Joe Burrow',
            'Daniels': 'Jayden Daniels',
            'Hurts': 'Jalen Hurts',
            'Conner': 'James Conner',
            'Wilson': 'Garrett Wilson',
            'Reed': 'Jayden Reed',
            'Addison': 'Jordan Addison',
            'Hall': 'Breece Hall',
            'Jackson': 'Lamar Jackson',
            'Johnson': 'Roschon Johnson',
            'Meyers': 'Jakobi Meyers',
            'Waddle': 'Jaylen Waddle',
            'Jeudy': 'Jerry Jeudy',
            'Mims Jr.': 'Marvin Mims Jr.',
            'Tolbert': 'Jalen Tolbert',
            'Coker': 'Jalen Coker',
            'Pearsall': 'Ricky Pearsall',
            'Mooney': 'Darnell Mooney',
            'Palmer': 'Joshua Palmer',
            'Nailor': 'Jalen Nailor',
        },
        'B.': {
            'Robinson': 'Bijan Robinson',
            'Irving': 'Bucky Irving',
            'Bowers': 'Brock Bowers',
            'Hall': 'Breece Hall',
            'Nix': 'Bo Nix',
            'Mayfield': 'Baker Mayfield',
            'Corum': 'Blake Corum',
            'Cooks': 'Brandin Cooks',
        },
        'C.': {
            'Lamb': 'CeeDee Lamb',
            'McCaffrey': 'Christian McCaffrey',
            'Brown': 'Chase Brown',
            'Hubbard': 'Chuba Hubbard',
            'Ridley': 'Calvin Ridley',
            'Sutton': 'Courtland Sutton',
            'Godwin': 'Chris Godwin',
            'Kupp': 'Cooper Kupp',
            'Olave': 'Chris Olave',
            'Samuel Sr.': 'Deebo Samuel Sr.',
        },
        'D.': {
            'Henry': 'Derrick Henry',
            'London': 'Drake London',
            'Achane': 'De\'Von Achane',
            'Adams': 'Davante Adams',
            'Metcalf': 'DK Metcalf',
            'Smith': 'DeVonta Smith',
            'Moore': 'DJ Moore',
            'Swift': 'D\'Andre Swift',
            'Montgomery': 'David Montgomery',
            'Diggs': 'Stefon Diggs',
            'Douglas': 'DeMario Douglas',
            'Mooney': 'Darnell Mooney',
            'Johnson': 'Diontae Johnson',
            'Hopkins': 'DeAndre Hopkins',
            'Slayton': 'Darius Slayton',
            'Vele': 'Devaughn Vele',
        },
        'A.': {
            'Jeanty': 'Ashton Jeanty',
            'Brown': 'A.J. Brown',
            'St. Brown': 'Amon-Ra St. Brown',
            'Kamara': 'Alvin Kamara',
            'Thielen': 'Adam Thielen',
            'Mitchell': 'Adonai Mitchell',
            'Cooper': 'Amari Cooper',
            'Jones': 'Aaron Jones Sr.',
        },
        'N.': {
            'Collins': 'Nico Collins',
            'Brown': 'Noah Brown',
            'Westbrook-Ikhine': 'Nick Westbrook-Ikhine',
        },
        'M.': {
            'Nabers': 'Malik Nabers',
            'Evans': 'Mike Evans',
            'Harrison Jr.': 'Marvin Harrison Jr.',
            'Wilson': 'Michael Wilson',
            'Marks': 'Woody Marks',
            'Lloyd': 'MarShawn Lloyd',
        },
        'P.': {
            'Nacua': 'Puka Nacua',
            'Pollard': 'Tony Pollard',
            'Pacheco': 'Isiah Pacheco',
            'Bryant': 'Pat Bryant',
        },
        'T.': {
            'Higgins': 'Tee Higgins',
            'McBride': 'Trey McBride',
            'McLaurin': 'Terry McLaurin',
            'Hill': 'Tyreek Hill',
            'Henderson': 'TreVeyon Henderson',
            'Hockenson': 'T.J. Hockenson',
            'Hunter': 'Travis Hunter',
            'Warren': 'Tyler Warren',
            'Johnson': 'Theo Johnson',
            'Tracy Jr.': 'Tyrone Tracy Jr.',
            'Harris': 'Tre Harris',
            'Tucker': 'Tre Tucker',
        },
        'G.': {
            'Kittle': 'George Kittle',
            'Wilson': 'Garrett Wilson',
            'Pickens': 'George Pickens',
            'Davis': 'Gabe Davis',
            'Dortch': 'Greg Dortch',
        },
        'K.': {
            'Walker III': 'Kenneth Walker III',
            'Williams': 'Kyren Williams',
            'Johnson': 'Kaleb Johnson',
            'Hunt': 'Kareem Hunt',
            'Gainwell': 'Kenneth Gainwell',
            'Mitchell': 'Keaton Mitchell',
            'Turpin': 'KaVontae Turpin',
        },
        'L.': {
            'Jackson': 'Lamar Jackson',
            'McConkey': 'Ladd McConkey',
            'Burden III': 'Luther Burden III',
        },
        'S.': {
            'Barkley': 'Saquon Barkley',
            'LaPorta': 'Sam LaPorta',
            'Diggs': 'Stefon Diggs',
            'Tucker': 'Sean Tucker',
        },
        'R.': {
            'Rice': 'Rashee Rice',
            'Odunze': 'Rome Odunze',
            'Harvey': 'RJ Harvey',
            'Stevenson': 'Rhamondre Stevenson',
            'Pearsall': 'Ricky Pearsall',
            'Ridley': 'Calvin Ridley',
            'Shaheed': 'Rashid Shaheed',
            'White': 'Rachaad White',
            'Davis': 'Ray Davis',
            'Robinson': 'Wan\'Dale Robinson',
            'Bateman': 'Rashod Bateman',
            'Dowdle': 'Rico Dowdle',
            'Johnson': 'Roschon Johnson',
            'Mostert': 'Raheem Mostert',
            'McCloud III': 'Ray-Ray McCloud III',
            'Wilson': 'Russell Wilson',
            'Rodgers': 'Aaron Rodgers',
        },
        'X.': {
            'Worthy': 'Xavier Worthy',
            'Legette': 'Xavier Legette',
        },
        'Z.': {
            'Flowers': 'Zay Flowers',
            'Charbonnet': 'Zach Charbonnet',
            'Ertz': 'Zach Ertz',
        },
        'W.': {
            'Robinson': 'Wan\'Dale Robinson',
            'Shipley': 'Will Shipley',
            'Marks': 'Woody Marks',
        },
        'E.': {
            'Egbuka': 'Emeka Egbuka',
            'Engram': 'Evan Engram',
            'Mitchell': 'Elijah Mitchell',
            'Edwards-Helaire': 'Clyde Edwards-Helaire',
            'Demercado': 'Emari Demercado',
            'Arroyo': 'Elijah Arroyo',
            'McPherson': 'Evan McPherson',
        },
        'I.': {
            'Pacheco': 'Isiah Pacheco',
            'Davis': 'Isaiah Davis',
            'Likely': 'Isaiah Likely',
            'Guerendo': 'Isaac Guerendo',
            'Iosivas': 'Andrei Iosivas',
            'Ingold': 'Alec Ingold',
        },
        'O.': {
            'Hampton': 'Omarion Hampton',
            'Gordon II': 'Ollie Gordon II',
            'Ogunbowale': 'Dare Ogunbowale',
        },
        'H.': {
            'Brown': 'Hollywood Brown',
            'Henry': 'Hunter Henry',
            'Butker': 'Harrison Butker',
            'Hopkins': 'DeAndre Hopkins',
            'Higbee': 'Tyler Higbee',
        },
        'V.': {
            'Jefferson': 'Van Jefferson',
            'Vele': 'Devaughn Vele',
        },
    }
    
    # Try to find in manual mapping
    if first_initial in first_name_map:
        if last_name in first_name_map[first_initial]:
            return first_name_map[first_initial][last_name]
    
    print(f"Warning: Could not find match for {abb_name} ({abb_team})")
    return abb_name

# Apply the mapping
adp['player_name'] = adp.apply(lambda row: find_full_name(row['Name'], row['Team_Clean'], all_players), axis=1)

# Clean up the Team column (remove bye week info) 
adp['team'] = adp['Team_Clean']

# Rename columns to match other files
adp_final = pd.DataFrame({
    'overall_rank': adp['RK'],
    'position': adp['POS.RK'].str.extract(r'([A-Z]+)')[0],
    'position_rank': adp['POS.RK'],
    'player_name': adp['player_name'],
    'team': adp['team'],
    'adp': adp['REAL_TIME_ADP']
})

# Save the standardized file
output_file = 'data/probability-models-draft/realtime_adp_20250822_standardized.csv'
adp_final.to_csv(output_file, index=False)
print(f"\nSaved standardized ADP file to: {output_file}")

# Show sample
print("\nFirst 20 rows of standardized file:")
print(adp_final.head(20))

# Show any unmatched names
unmatched = adp_final[adp_final['player_name'].str.contains(r'^[A-Z]\.', na=False)]
if not unmatched.empty:
    print(f"\nWarning: {len(unmatched)} names could not be matched:")
    print(unmatched[['player_name', 'team']].head(20))