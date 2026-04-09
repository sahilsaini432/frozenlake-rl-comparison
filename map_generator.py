"""Standalone FrozenLake map generator.

AI Generated script to create custom FrozenLake maps with multiple paths and moderate hole density.

Run this script directly to generate maps and save them to maps.json:
    python3 helper/map_generator.py

Design goals:
  - Multiple viable paths from S (top-left) to G (bottom-right)
  - Wide corridors (2-3 tiles) so the agent can recover in slippery mode
  - Moderate hole density (~15%) to create interesting decisions without blocking all routes
"""

import argparse
import json
import random
from collections import deque
from os import path


def generate_map(size, seed=None, hole_density=0.15):
    """Generate a size x size FrozenLake map with multiple paths to the goal."""
    if seed is None:
        seed = size
    rng = random.Random(seed)

    grid = [["F"] * size for _ in range(size)]
    grid[0][0] = "S"
    grid[size - 1][size - 1] = "G"

    # Protected zone around S and G — no holes within Manhattan distance 2
    def is_protected(r, c):
        return (r + c <= 2) or ((size - 1 - r) + (size - 1 - c) <= 2)

    # Place holes randomly, then verify connectivity
    max_attempts = 50
    for _ in range(max_attempts):
        for r in range(size):
            for c in range(size):
                if grid[r][c] == "H":
                    grid[r][c] = "F"

        for r in range(size):
            for c in range(size):
                if grid[r][c] in ("S", "G"):
                    continue
                if is_protected(r, c):
                    continue
                if rng.random() < hole_density:
                    grid[r][c] = "H"

        if _count_disjoint_paths(grid, size) >= 2:
            break

    return ["".join(row) for row in grid]


def _count_disjoint_paths(grid, size):
    """Count node-disjoint paths from S to G using iterative BFS augmentation."""
    start = (0, 0)
    goal = (size - 1, size - 1)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    blocked = set()
    count = 0

    for _ in range(3):
        path = _bfs_path(grid, size, start, goal, blocked, directions)
        if path is None:
            break
        count += 1
        for node in path[1:-1]:
            blocked.add(node)

    return count


def _bfs_path(grid, size, start, goal, blocked, directions):
    """BFS shortest path avoiding holes and blocked nodes."""
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (r, c), p = queue.popleft()
        if (r, c) == goal:
            return p
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if (nr, nc) not in visited and (nr, nc) not in blocked and grid[nr][nc] != "H":
                    visited.add((nr, nc))
                    queue.append(((nr, nc), p + [(nr, nc)]))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FrozenLake maps and save to JSON")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--density", type=float, default=0.15, help="Hole density (default: 0.15)")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Grid sizes to generate (default: 16 32 64)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=path.join(path.dirname(__file__), "..", "maps.json"),
        help="Output file path (default: MCTS/maps.json)",
    )
    args = parser.parse_args()

    output = path.abspath(args.output)

    # Load existing maps so we append rather than overwrite
    existing = {}
    if path.exists(output):
        with open(output) as f:
            existing = json.load(f)

    for size in args.sizes:
        m = generate_map(size, seed=args.seed, hole_density=args.density)
        grid = [list(row) for row in m]
        holes = sum(row.count("H") for row in m)
        paths = _count_disjoint_paths(grid, size)
        print(
            f"{size}x{size}: {holes}/{size*size} holes ({holes/(size*size)*100:.1f}%), {paths} disjoint paths"
        )
        existing[str(size)] = m

    with open(output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {output}")
