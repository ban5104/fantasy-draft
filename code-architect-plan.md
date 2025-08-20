  Core approach (from feedback.md):
  - DP-over-positions with only 48
  states - globally optimal and
  computationally tiny
  - Ladder EV rewards using survival
  probabilities
  - Backward induction: F(k,r,w,q,t) =
   max{ladder_ev + F(k+1,...)}

  Implementation details (from 
  plan.md):
  - Single file: dp_draft_optimizer.py
   (~200 lines)
  - Minimal dependencies: pandas,
  numpy, scipy, rapidfuzz
  - 10 Monte Carlo simulations for
  development speed
  - Console output only - no
  visualization

  The structure:
  1. Data loading (30 lines) - Load
  CSVs, fuzzy-match player names
  2. Monte Carlo (40 lines) - Compute
  survival probabilities
  3. Ladder EV (30 lines) - Expected
  value calculations
  4. DP solver (50 lines) - Backward
  induction over 48 states
  5. Main execution (50 lines) - Snake
   draft setup, results output

  This MVP validates your mathematical
   theory that combining survival
  probabilities with dynamic
  programming over positions (not
  players) produces an optimal draft
  strategy. Once this works, you can
  scale up the simulations and add
  more sophisticated features.