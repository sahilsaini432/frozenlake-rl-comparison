# MCTS for FrozenLake

> **Disclaimer:** This document was generated with the assistance of Claude (Anthropic AI) to explain how the MCTS algorithm structure is setup.

Run from the `MCTS/` directory:
```
python3 run_mcts.py [options]
```

---

## CLI Arguments

| Flag | Default | Choices / Type | Description |
|------|---------|----------------|-------------|
| `-e`, `--episodes` | `100` | `int` | Number of evaluation episodes |
| `-g`, `--grid` | `4` | `4 8 16 32 64` | Grid size for FrozenLake |
| `-s`, `--slip` | `False` | flag | Enable slippery (stochastic) environment |
| `-c`, `--exploration_constant` | `1.4` | `float` | UCT/PUCT exploration constant C |
| `--selection` | `uct` | see below | Tree node selection strategy |
| `--rollout` | `random` | see below | Rollout/simulation policy |
| `--final_action` | `robust_child` | see below | How to pick the final action |
| `--backprop` | `standard` | `standard max` | Backpropagation strategy |
| `--expansion` | `standard` | see below | Strategy for tree expansion |
| `-v`, `--verbose` | `False` | flag | Verbose per-step logging |
| `--human` | `False` | flag | Render with human-visible GUI |


---

## Strategy Options

### Selection (`--selection`)
| Value | Class | Description |
|-------|-------|-------------|
| `uct` | `UCTStrategy` | Upper Confidence Bounds for Trees |
| `ucb1` | `UCB1Strategy` | Classic UCB1 (no tree-specific adjustment) |
| `puct_uniform` | `PUCTStrategy_Uniform` | PUCT with uniform prior (1/num_actions) |
| `puct_heuristic` | `PUCTStrategy_Heuristic` | PUCT with heuristic-based prior |
| `puct_softmax` | `PUCTStrategy_Softmax` | PUCT with softmax-derived prior (set lazily) |

### Rollout (`--rollout`)
| Value | Class | Description |
|-------|-------|-------------|
| `random` | `RandomRollout` | Uniform random rollout |
| `epsilon_greedy` | `EpsilonGreedyRollout` | ε-greedy rollout (ε=0.1) |
| `value_network` | `ValueNetworkRollout` | Blend random rollout + heuristic value function |
| `mlp_value_network` | `ValueNetworkRollout` | Blend random rollout + trained MLP (runs VI + training first) |
| `alphazero` | `ValueFunctionOnly` | No rollout — use trained MLP value only (AlphaZero-style) |

> `mlp_value_network` and `alphazero` train a value network via value iteration before evaluation starts (slow first run).

### Final Action (`--final_action`)
| Value | Class | Description |
|-------|-------|-------------|
| `robust_child` | `RobustChild` | Pick the most-visited child |
| `max_value` | `MaxValue` | Pick the highest-value child |
| `softmax_visits` | `SoftmaxVisits` | Sample from softmax over visit counts (T=1.0) |

### Backpropagation (`--backprop`)
| Value | Class | Description |
|-------|-------|-------------|
| `standard` | `StandardBackprop` | Average the returns up the tree |
| `max` | `MaxBackprop` | Propagate the max return seen |

### Expansion (`--expansion`)
| Value | Class | Description |
|-------|-------|-------------|
| `standard` | `StandardExpansion` | Expand one unvisited child at a time |
| `full` | `FullExpansion` | Expand all children of a node at once |
| `progressive_widening` | `ProgressiveWideningExpansion` | Limit children added proportional to visit count (controlled by `α`) |

### Simulation counts (auto-scaled by grid size)
| Grid | Simulations | Rollout depth |
|------|-------------|---------------|
| 4×4 | 1 000 | 100 |
| 8×8 | 3 000 | 200 |
| 16×16 | 8 000 | 400 |
| 32×32 | 15 000 | 800 |
| 64×64 | 30 000 | 1 600 |

---

## Example Commands

### Baseline — UCT + random rollout, 4×4
```bash
python3 run_mcts.py --selection uct --rollout random --final_action robust_child -g 4 -e 100
```

### UCB1, non-slippery, 8×8
```bash
python3 run_mcts.py --selection ucb1 --rollout random -g 8 -e 100
```

### PUCT (uniform prior) + epsilon-greedy rollout
```bash
python3 run_mcts.py --selection puct_uniform --rollout epsilon_greedy -g 4 -e 100
```

### PUCT (heuristic prior) + heuristic value network blend
```bash
python3 run_mcts.py --selection puct_heuristic --rollout value_network -g 4 -e 100
```

### PUCT (softmax prior) + heuristic value network blend
```bash
python3 run_mcts.py --selection puct_softmax --rollout value_network -g 4 -e 100
```

### AlphaZero-style (trained MLP, no rollout)
```bash
python3 run_mcts.py --selection puct_uniform --rollout alphazero --final_action robust_child -g 4 -e 100
```

### MLP value network + random rollout blend
```bash
python3 run_mcts.py --selection uct --rollout mlp_value_network -g 4 -e 100
```

### Max backpropagation
```bash
python3 run_mcts.py --selection uct --rollout random --backprop max -g 4 -e 100
```

### Softmax final action
```bash
python3 run_mcts.py --selection uct --rollout random --final_action softmax_visits -g 4 -e 100
```

### Slippery environment, 8×8
```bash
python3 run_mcts.py --selection uct --rollout random -g 8 -s -e 100
```

### Large grid — 16×16
```bash
python3 run_mcts.py --selection uct --rollout random -g 16 -e 50
```

### Large grid — 32×32
```bash
python3 run_mcts.py --selection uct --rollout random -g 32 -e 20
```

### High exploration constant
```bash
python3 run_mcts.py --selection uct --rollout random -c 2.0 -g 4 -e 100
```

### Low exploration constant
```bash
python3 run_mcts.py --selection uct --rollout random -c 0.5 -g 4 -e 100
```

---

## Run Checklist

Different Configurations to test

---

### 1. Selection × Rollout
_Fixed: `--final_action robust_child --backprop standard -g 8 -e 100`_

#### uct
- [x] `uct` + `random` — `python3 run_mcts.py --selection uct --rollout random --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `uct` + `epsilon_greedy` — `python3 run_mcts.py --selection uct --rollout epsilon_greedy --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `uct` + `value_network` — `python3 run_mcts.py --selection uct --rollout value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `uct` + `mlp_value_network` — `python3 run_mcts.py --selection uct --rollout mlp_value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `uct` + `alphazero` — `python3 run_mcts.py --selection uct --rollout alphazero --final_action robust_child --backprop standard -g 8 -e 100`

#### ucb1
- [x] `ucb1` + `random` — `python3 run_mcts.py --selection ucb1 --rollout random --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `ucb1` + `epsilon_greedy` — `python3 run_mcts.py --selection ucb1 --rollout epsilon_greedy --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `ucb1` + `value_network` — `python3 run_mcts.py --selection ucb1 --rollout value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `ucb1` + `mlp_value_network` — `python3 run_mcts.py --selection ucb1 --rollout mlp_value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `ucb1` + `alphazero` — `python3 run_mcts.py --selection ucb1 --rollout alphazero --final_action robust_child --backprop standard -g 8 -e 100`

#### puct_uniform
- [x] `puct_uniform` + `random` — `python3 run_mcts.py --selection puct_uniform --rollout random --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_uniform` + `epsilon_greedy` — `python3 run_mcts.py --selection puct_uniform --rollout epsilon_greedy --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_uniform` + `value_network` — `python3 run_mcts.py --selection puct_uniform --rollout value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_uniform` + `mlp_value_network` — `python3 run_mcts.py --selection puct_uniform --rollout mlp_value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_uniform` + `alphazero` — `python3 run_mcts.py --selection puct_uniform --rollout alphazero --final_action robust_child --backprop standard -g 8 -e 100`

#### puct_heuristic
- [x] `puct_heuristic` + `random` — `python3 run_mcts.py --selection puct_heuristic --rollout random --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_heuristic` + `epsilon_greedy` — `python3 run_mcts.py --selection puct_heuristic --rollout epsilon_greedy --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_heuristic` + `value_network` — `python3 run_mcts.py --selection puct_heuristic --rollout value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_heuristic` + `mlp_value_network` — `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_heuristic` + `alphazero` — `python3 run_mcts.py --selection puct_heuristic --rollout alphazero --final_action robust_child --backprop standard -g 8 -e 100`

#### puct_softmax
- [x] `puct_softmax` + `random` — `python3 run_mcts.py --selection puct_softmax --rollout random --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_softmax` + `epsilon_greedy` — `python3 run_mcts.py --selection puct_softmax --rollout epsilon_greedy --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_softmax` + `value_network` — `python3 run_mcts.py --selection puct_softmax --rollout value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_softmax` + `mlp_value_network` — `python3 run_mcts.py --selection puct_softmax --rollout mlp_value_network --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `puct_softmax` + `alphazero` — `python3 run_mcts.py --selection puct_softmax --rollout alphazero --final_action robust_child --backprop standard -g 8 -e 100`

---

### 2. Final Action × Backpropagation
_Fixed: `--selection uct --rollout random -g 4 -e 100`_

#### robust_child
- [x] `robust_child` + `standard` — `python3 run_mcts.py --selection uct --rollout random --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `robust_child` + `max` — `python3 run_mcts.py --selection uct --rollout random --final_action robust_child --backprop max -g 8 -e 100`

#### max_value
- [x] `max_value` + `standard` — `python3 run_mcts.py --selection uct --rollout random --final_action max_value --backprop standard -g 8 -e 100`
- [x] `max_value` + `max` — `python3 run_mcts.py --selection uct --rollout random --final_action max_value --backprop max -g 8 -e 100`

#### softmax_visits
- [x] `softmax_visits` + `standard` — `python3 run_mcts.py --selection uct --rollout random --final_action softmax_visits --backprop standard -g 8 -e 100`
- [x] `softmax_visits` + `max` — `python3 run_mcts.py --selection uct --rollout random --final_action softmax_visits --backprop max -g 8 -e 100`

---

### 3. Grid Size — Non-Slippery
_Fixed: `--selection uct --rollout random --final_action robust_child --backprop standard`_

- [ ] 4×4 — `python3 run_mcts.py --selection uct --rollout random -g 4 -e 100`
- [ ] 8×8 — `python3 run_mcts.py --selection uct --rollout random -g 8 -e 100`
- [ ] 16×16 — `python3 run_mcts.py --selection uct --rollout random -g 16 -e 50`
- [ ] 32×32 — `python3 run_mcts.py --selection uct --rollout random -g 32 -e 20`
- [ ] 64×64 — `python3 run_mcts.py --selection uct --rollout random -g 64 -e 10`

---

### 4. Grid Size — Slippery
_Fixed: `--selection uct --rollout random --final_action robust_child --backprop standard -s`_

- [ ] 4×4 slippery — `python3 run_mcts.py --selection uct --rollout random -g 4 -s -e 100`
- [ ] 8×8 slippery — `python3 run_mcts.py --selection uct --rollout random -g 8 -s -e 100`
- [ ] 16×16 slippery — `python3 run_mcts.py --selection uct --rollout random -g 16 -s -e 50`
- [ ] 32×32 slippery — `python3 run_mcts.py --selection uct --rollout random -g 32 -s -e 20`
- [ ] 64×64 slippery — `python3 run_mcts.py --selection uct --rollout random -g 64 -s -e 10`

---

### 5. Exploration Constant Sweep
_Fixed: `--selection uct --rollout random --final_action robust_child --backprop standard -g 4 -e 100`_

- [ ] C=0.5 — `python3 run_mcts.py --selection uct --rollout random -c 0.5 -g 4 -e 100`
- [ ] C=1.0 — `python3 run_mcts.py --selection uct --rollout random -c 1.0 -g 4 -e 100`
- [ ] C=1.4 (default) — `python3 run_mcts.py --selection uct --rollout random -c 1.4 -g 4 -e 100`
- [ ] C=2.0 — `python3 run_mcts.py --selection uct --rollout random -c 2.0 -g 4 -e 100`
- [ ] C=3.0 — `python3 run_mcts.py --selection uct --rollout random -c 3.0 -g 4 -e 100`

---

### 6. Recommended Combinations

These are the configurations most likely to yield strong results, based on how well each component's strengths complement the others.

#### AlphaZero-style — no rollout, pure learned value
_Why: eliminates rollout variance entirely; the trained MLP covers the role of both rollout and evaluation. The most principled modern approach._
- [x] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --final_action robust_child --backprop standard -g 4 -e 100`

#### Fully heuristic-guided — informed selection + informed evaluation
_Why: both the selection prior (`puct_heuristic`) and the rollout blend (`value_network`) use domain knowledge about the grid. Best ratio of performance to cost on small/medium grids._
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --final_action robust_child --backprop standard -g 4 -e 100`
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --final_action robust_child --backprop standard -g 8 -e 100`

#### Trained prior + trained value — best of both
_Why: `puct_heuristic` sets a strong prior at selection time, while `mlp_value_network` replaces noisy random rollouts with a learned value estimate. Strongest expected accuracy, at the cost of upfront training._
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --final_action robust_child --backprop standard -g 4 -e 100`

#### Max-path for deterministic environments
_Why: in non-slippery FrozenLake there is a fixed best path. `max` backprop propagates the best return seen (not the average), and `max_value` final action exploits it. Effective when the environment has low stochasticity._
- [x] `python3 run_mcts.py --selection uct --rollout random --final_action max_value --backprop max -g 8 -e 100`

#### Best for slippery (stochastic) environments
_Why: `epsilon_greedy` rollout approximates a near-optimal stochastic policy better than pure random. `robust_child` + `standard` backprop are robust to outcome variance, which is high under slipperiness._
- [x]  `python3 run_mcts.py --selection uct --rollout epsilon_greedy --final_action robust_child --backprop standard -g 8 -s -e 100`

#### Softmax prior + trained value on larger grid
_Why: `puct_softmax` sets priors lazily from value estimates, pairing well with `mlp_value_network`. Scales better to larger grids where random rollouts are too shallow to return reliable signal._
- [x] `python3 run_mcts.py --selection puct_softmax --rollout mlp_value_network --final_action robust_child --backprop standard -g 8 -e 100`

#### AlphaZero-style on 8×8
_Why: same reasoning as the 4×4 AlphaZero entry, but tests whether the trained value generalises to a harder grid where rollout-based methods degrade._
- [x] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --final_action robust_child --backprop standard -g 8 -e 100`

#### Heuristic-guided on slippery 8×8
_Why: combining domain-aware selection and evaluation with a harder stochastic grid is a strong stress-test. Expected to outperform UCT+random on this setting by a meaningful margin._
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --final_action robust_child --backprop standard -g 8 -s -e 100`

#### Progressive widening baseline — controlled branching vs. standard expansion
_Why: progressive widening (α=0.5) limits children added to N^α, forcing the tree to visit existing nodes more before expanding new ones. Direct apples-to-apples comparison against `standard` expansion on the same rollout reveals whether focused depth beats breadth._
- [x] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 8 -e 100`

#### Progressive widening + heuristic prior — best focus strategy on medium grids
_Why: `puct_heuristic` priors steer *which* children get expanded, while progressive widening controls *how many* are opened at each visit count. Together they concentrate simulations on the most promising narrow subtree, especially valuable on 8×8 where the branching factor can dilute visits._
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 8 -e 100`
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 8 -e 100`

#### AlphaZero-style + progressive widening
_Why: with no rollout the tree relies entirely on the MLP value; progressive widening prevents the search from prematurely expanding too many children before their value estimates are reliable. Mirrors how AlphaZero's PUCT naturally regulates breadth._
- [x] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 8 -e 100`

#### Progressive widening on slippery environment
_Why: stochastic transitions increase outcome variance per node visit, making it beneficial to deepen existing nodes (more samples of the same transition) rather than expanding new children. Progressive widening's depth-first bias should help here relative to standard expansion._
- [x] `python3 run_mcts.py --selection uct --rollout epsilon_greedy --expansion progressive_widening --final_action robust_child --backprop standard -g 8 -s -e 100`
- [x] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 8 -s -e 100`

---

### 7. Large Grid Configurations

Episode counts are reduced as grid size grows to keep runtime manageable. `value_network` and `alphazero` are preferred over `random` rollout on large grids because random rollouts rarely reach the goal and return no useful signal. `progressive_widening` expansion is used throughout — on large grids, expanding all children at once wastes simulation budget on unvisited branches; widening focuses depth on the most promising children first.

#### 16×16 — non-slippery
- [ ] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -e 50`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -e 50`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -e 50`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -e 50`

#### 16×16 — slippery
- [ ] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -s -e 50`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -s -e 50`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 16 -s -e 50`

#### 32×32 — non-slippery
- [ ] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -e 20`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -e 20`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -e 20`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -e 20`

#### 32×32 — slippery
- [ ] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -s -e 20`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -s -e 20`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 32 -s -e 20`

#### 64×64 — non-slippery
- [ ] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -e 10`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -e 10`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -e 10`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -e 10`

#### 64×64 — slippery
- [ ] `python3 run_mcts.py --selection uct --rollout random --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -s -e 10`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -s -e 10`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 64 -s -e 10`

#### 128×128 — non-slippery
_Random rollout is essentially useless at this scale — `alphazero` and `value_network` are the only viable options._
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 128 -e 5`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 128 -e 5`
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout mlp_value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 128 -e 5`

#### 128×128 — slippery
- [ ] `python3 run_mcts.py --selection puct_heuristic --rollout value_network --expansion progressive_widening --final_action robust_child --backprop standard -g 128 -s -e 5`
- [ ] `python3 run_mcts.py --selection puct_uniform --rollout alphazero --expansion progressive_widening --final_action robust_child --backprop standard -g 128 -s -e 5`
