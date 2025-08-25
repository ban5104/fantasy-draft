# Fantasy Draft Optimizer Enhancement Plan

## Overview
Transform the existing draft optimizer from a single-answer tool into a comprehensive draft-day decision support system. This plan implements the most valuable enhancements while preserving the unified architecture of `scripts/dp_draft_optimizer_debug.py`.

## Implementation Priority

### Phase 1: Core Enhancements (High Impact, Medium Effort)

#### 1. ε-Optimal Plans Menu ✅
**Goal**: Show multiple near-optimal draft strategies instead of just one.

**Implementation**:
- Modify `dp_optimize()` function (line 976) to use K-best dynamic programming
- Return top-K sequences within ε% (default 1-2%) of optimal EV
- Use beam search approach with `heapq.nlargest(K, candidates)`

**Code Changes**:
```python
# Replace single best tracking with K-best heap
from heapq import nlargest

@lru_cache(maxsize=None)
def dp_optimize_k_best(pick_idx, rb_count, wr_count, qb_count, te_count, k=8):
    # ... existing validation ...
    
    candidates = []  # (total_value, position, details)
    
    for pos in ["RB", "WR", "QB", "TE"]:
        if counts[pos] < POSITION_LIMITS[pos]:
            # ... existing EV calculation ...
            candidates.append((total_value, pos, {'ev': ev, 'future': future_value}))
    
    # Return top-K instead of just best
    top_k = nlargest(k, candidates, key=lambda x: x[0])
    return top_k

def get_epsilon_optimal_plans(epsilon=0.02):
    """Return all draft plans within epsilon of optimal."""
    all_plans = dp_optimize_k_best(0, 0, 0, 0, 0)
    best_value = all_plans[0][0]
    threshold = best_value * (1 - epsilon)
    return [plan for plan in all_plans if plan[0] >= threshold]
```

**Output Enhancement**:
```
=== DRAFT PLAN MENU (ε=1.5%) ===
Plan A: RB-QB-RB-WR-RB-WR-TE    EV 1380.9  (baseline)
Plan B: RB-WR-RB-WR-RB-QB-TE    EV 1374.1  (-0.5%, safer)
Plan C: WR-RB-QB-WR-RB-WR-TE    EV 1370.2  (-0.8%, WR heavy)
```

#### 2. Regret Tables Per Pick ✅
**Goal**: Show "what if" scenarios for each draft pick.

**Implementation**:
- Leverage existing `ladder_ev_debug()` function (line 823)
- For each pick, compute ΔEV for top 3-5 alternative positions
- Add to `show_pick_analysis()` function (line 914)

**Code Changes**:
```python
def compute_pick_regret(pick_idx, current_counts):
    """Compute regret for alternative choices at this pick."""
    pick_number = SNAKE_PICKS[pick_idx]
    regret_table = []
    
    # Get current optimal choice
    optimal_value, optimal_pos = dp_optimize(pick_idx, **current_counts)
    
    # Evaluate each alternative
    for alt_pos in ["RB", "WR", "QB", "TE"]:
        if current_counts[alt_pos] < POSITION_LIMITS[alt_pos]:
            # Force this position choice
            alt_counts = current_counts.copy()
            alt_counts[alt_pos] += 1
            
            # Get immediate EV
            slot = current_counts[alt_pos] + 1
            immediate_ev, _ = ladder_ev_debug(alt_pos, pick_number, slot, PLAYERS, SURVIVAL_PROBS)
            
            # Get future EV
            future_ev, _ = dp_optimize(pick_idx + 1, **alt_counts)
            total_alt_ev = immediate_ev + future_ev
            
            regret = optimal_value - total_alt_ev
            regret_table.append({
                'position': alt_pos,
                'ev': total_alt_ev,
                'regret': regret,
                'regret_pct': (regret / optimal_value) * 100
            })
    
    return sorted(regret_table, key=lambda x: x['regret'])
```

#### 3. Time-to-Cliff Analysis Enhancement ✅  
**Goal**: Extract cliff warnings from existing value decay analysis.

**Implementation**:
- Enhance existing `show_value_decay_analysis()` (line 1512)
- Add "safe window" calculations
- Surface picks-to-cliff for each position

**Code Changes**:
```python
def compute_cliff_windows(pick_idx, counts, cliff_threshold=0.15):
    """Compute how many picks until each position hits value cliff."""
    current_pick = SNAKE_PICKS[pick_idx]
    remaining_picks = SNAKE_PICKS[pick_idx:]
    
    windows = {}
    for pos in ["RB", "WR", "QB", "TE"]:
        if counts[pos] < POSITION_LIMITS[pos]:
            windows[pos] = {
                'picks_to_cliff': None,
                'safe_window': 0,
                'cliff_drop_pct': 0
            }
            
            pos_players = pos_sorted(PLAYERS, pos)
            survival_matrix = SURVIVAL_PROBS.get(pos)
            
            if survival_matrix is not None:
                # Find best available now
                current_best_ev = get_best_available_ev(pos, current_pick, survival_matrix, pos_players)
                
                # Check each future pick for cliff
                for i, future_pick in enumerate(remaining_picks[1:], 1):
                    future_ev = get_best_available_ev(pos, future_pick, survival_matrix, pos_players)
                    
                    if current_best_ev > 0:
                        drop_pct = (current_best_ev - future_ev) / current_best_ev
                        if drop_pct >= cliff_threshold:
                            windows[pos]['picks_to_cliff'] = i
                            windows[pos]['cliff_drop_pct'] = drop_pct * 100
                            break
                        elif drop_pct < 0.10:  # Safe threshold
                            windows[pos]['safe_window'] = i
    
    return windows
```

### Phase 2: Analysis Enhancements (High Impact, Low Effort)

#### 4. Flexibility Index ✅
**Goal**: Quantify how flexible each pick decision is.

**Implementation**:
```python
import numpy as np

def compute_flexibility_index(position_values):
    """Compute entropy-based flexibility score."""
    if not position_values:
        return 0.0
    
    evs = [v['total_value'] for v in position_values.values()]
    if len(evs) <= 1:
        return 0.0
    
    # Normalize to probabilities
    ev_array = np.array(evs)
    ev_array = ev_array - ev_array.min() + 1  # Shift to positive
    probs = ev_array / ev_array.sum()
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(len(probs))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0

# Add to show_pick_analysis()
flexibility = compute_flexibility_index(position_values)
print(f"Pick {pick_number} Flexibility Index: {flexibility:.2f} {'(many options)' if flexibility > 0.7 else '(limited options)' if flexibility < 0.3 else '(moderate options)'}")
```

#### 5. Contingency Trees ✅
**Goal**: Build "if-then" decision trees for each pick.

**Implementation**:
```python
def build_contingency_tree(pick_idx, counts, max_depth=2):
    """Build decision tree for pick contingencies."""
    pick_number = SNAKE_PICKS[pick_idx]
    
    # Primary recommendation
    primary_value, primary_pos = dp_optimize(pick_idx, **counts)
    
    # Get top players for primary position
    primary_players = get_top_available_players(primary_pos, pick_number, n=3)
    
    # If primary unavailable, what's next best?
    regret_table = compute_pick_regret(pick_idx, counts)
    secondary_pos = regret_table[1]['position'] if len(regret_table) > 1 else None
    secondary_players = get_top_available_players(secondary_pos, pick_number, n=2) if secondary_pos else []
    
    # If both unavailable
    tertiary_pos = regret_table[2]['position'] if len(regret_table) > 2 else None
    tertiary_players = get_top_available_players(tertiary_pos, pick_number, n=2) if tertiary_pos else []
    
    return {
        'primary': {'position': primary_pos, 'players': primary_players, 'ev': primary_value},
        'secondary': {'position': secondary_pos, 'players': secondary_players, 'regret_pct': regret_table[1]['regret_pct'] if len(regret_table) > 1 else 0},
        'tertiary': {'position': tertiary_pos, 'players': tertiary_players, 'regret_pct': regret_table[2]['regret_pct'] if len(regret_table) > 2 else 0}
    }

def show_contingency_tree(tree, pick_number):
    """Display the contingency decision tree."""
    print(f"\nCONTINGENCY PLAYBOOK (Pick {pick_number}):")
    print(f"🎯 PRIMARY: {tree['primary']['position']} → {', '.join([p.name for p in tree['primary']['players'][:2]])}")
    
    if tree['secondary']['position']:
        print(f"📋 IF GONE: {tree['secondary']['position']} → {', '.join([p.name for p in tree['secondary']['players'][:2]])} (-{tree['secondary']['regret_pct']:.1f}% EV)")
    
    if tree['tertiary']['position']:
        print(f"🔄 LAST RESORT: {tree['tertiary']['position']} → {', '.join([p.name for p in tree['tertiary']['players'][:2]])} (-{tree['tertiary']['regret_pct']:.1f}% EV)")
```

### Phase 3: Risk & Robustness (Medium Impact, Medium Effort)

#### 6. Risk-Aware Variants ✅
**Goal**: Provide floor-focused and upside-focused alternatives using envelope data.

**Implementation**:
```python
def compute_risk_variants(players, survival_probs):
    """Compute floor-focused and upside-focused draft plans."""
    if not USE_ENVELOPES:
        return None
    
    variants = {}
    
    # Floor-focused: maximize CVaR (5th percentile)
    def floor_score(player):
        low = getattr(player, 'low', player.points)
        return low * 0.8 + player.points * 0.2  # Weight floor heavily
    
    # Upside-focused: maximize ceiling potential  
    def upside_score(player):
        high = getattr(player, 'high', player.points)
        return high * 0.8 + player.points * 0.2  # Weight ceiling heavily
    
    # Compute alternative sequences (simplified)
    variants['floor'] = optimize_with_custom_scoring(floor_score)
    variants['upside'] = optimize_with_custom_scoring(upside_score)
    
    return variants
```

#### 7. Robustness Analysis ✅
**Goal**: Test stability across different data sources.

**Implementation**:
```python
def run_robustness_analysis():
    """Run optimizer across multiple data sources."""
    data_sources = [
        'espn_projections_20250814.csv',
        'espn_algorithm_20250824.csv'
    ]
    
    results = {}
    for source in data_sources:
        if os.path.exists(f'data/probability-models-draft/{source}'):
            # Run optimization with this data source
            players = load_and_merge_data(f'data/probability-models-draft/{source}')
            player_survival = monte_carlo_survival_realistic(players, 1000)
            SURVIVAL_PROBS = get_position_survival_matrix(players, player_survival)
            
            # Get optimal sequence
            sequence = get_optimal_sequence()
            results[source] = sequence
    
    # Analyze consensus
    return analyze_consensus(results)
```

## Integration Points

### Modified Functions
- `dp_optimize()` → `dp_optimize_k_best()` for ε-optimal plans
- `show_pick_analysis()` → Enhanced with regret tables and flexibility
- `show_value_decay_analysis()` → Enhanced with cliff windows
- `main()` → New output sections for plan menu and contingencies

### New Functions to Add
- `get_epsilon_optimal_plans()`
- `compute_pick_regret()`
- `compute_flexibility_index()`
- `build_contingency_tree()`
- `compute_cliff_windows()`
- `show_contingency_tree()`
- `show_plan_menu()`

### New Output Sections
1. **Plan Menu** (after current optimal strategy)
2. **Per-Pick Enhancements** (in existing pick analysis)
   - Regret table
   - Flexibility index
   - Cliff windows
   - Contingency tree
3. **Risk Variants** (optional, if envelope data available)
4. **Robustness Matrix** (if multiple data sources)

## Expected Output Enhancement

```
=== DRAFT PLAN MENU (ε=1.5%) ===
Plan A: RB-QB-RB-WR-RB-WR-TE    EV 1380.9  (baseline)
Plan B: RB-WR-RB-WR-RB-QB-TE    EV 1374.1  (-0.5%, safer)  
Plan C: WR-RB-QB-WR-RB-WR-TE    EV 1370.2  (-0.8%, WR heavy)

PICK 5 ANALYSIS 
Flexibility Index: 0.71 (many viable options)
Windows: RB cliff in 1 pick, WR safe for 2 picks, TE safe for 3 picks

REGRET TABLE:
Position    EV     Regret    Notes
RB        380.9    0.0%     (optimal)
WR        375.2   -1.5%     Strong alternative
QB        365.1   -4.2%     Early but viable
TE        340.8  -10.6%     Significant drop

CONTINGENCY PLAYBOOK:
🎯 PRIMARY: RB → Saquon Barkley, Jonathan Taylor
📋 IF GONE: WR → CeeDee Lamb, Tyreek Hill (-1.5% EV)
🔄 LAST RESORT: QB → Josh Allen, Lamar Jackson (-4.2% EV)
```

## Testing Integration

Enhance existing `test_optimizer.py` to validate new features:

```python
def test_enhanced_features():
    """Test new enhancement features."""
    
    # Test ε-optimal plans
    plans = get_epsilon_optimal_plans(epsilon=0.02)
    assert len(plans) >= 1
    assert plans[0][0] >= plans[1][0]  # Sorted by EV
    
    # Test regret analysis
    regret = compute_pick_regret(0, {"RB": 0, "WR": 0, "QB": 0, "TE": 0})
    assert len(regret) >= 2
    assert regret[0]['regret'] <= regret[1]['regret']
    
    # Test flexibility calculation
    flex = compute_flexibility_index({"RB": {"total_value": 100}, "WR": {"total_value": 95}})
    assert 0 <= flex <= 1
    
    print("✅ All enhanced features working correctly")
```

## Implementation Notes

- All enhancements preserve the unified script architecture
- Existing functions are enhanced, not replaced
- New features are additive and optional
- Analytics capture integration with existing envelope system
- Backward compatibility maintained for existing usage patterns

## Success Criteria

✅ Multiple draft plans within ε% of optimal  
✅ Regret analysis for alternative picks  
✅ Time-to-cliff warnings  
✅ Flexibility scoring per pick  
✅ Contingency decision trees  
✅ Enhanced testing validation  
✅ Clean integration with existing codebase  
✅ Maintains unified architecture constraint