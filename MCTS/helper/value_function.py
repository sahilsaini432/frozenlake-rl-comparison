import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class HeuristicValueFunction:
    """Callable wrapper for the heuristic value function."""

    def __init__(self, grid_size):
        self.value_fn = self.make_heuristic_value_fn(grid_size)

    def __call__(self, state) -> float:
        return self.value_fn(state)

    def make_heuristic_value_fn(self, grid_size):
        """
        V(s) = -(Manhattan distance to goal) / max_possible_distance, in [-1, 0].
        Terminal goal state gets 1.0; all other states are negative proportional to distance.
        """
        goal = grid_size * grid_size - 1
        goal_row, goal_col = divmod(goal, grid_size)
        max_dist = (grid_size - 1) + (grid_size - 1)

        def value_fn(state: int) -> float:
            if state == goal:
                return 1.0
            row, col = divmod(state, grid_size)
            dist = abs(row - goal_row) + abs(col - goal_col)
            return -dist / max_dist

        return value_fn


class ValueMLP(nn.Module):
    """Small MLP: (row, col, dist_to_goal) -> V(s)."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def value_iteration(self, env, gamma=0.99, theta=1e-8, max_iter=10000):
        """Create value table V(s) using Value Iteration. Returns a numpy array of shape (n_states,)."""
        P = env.unwrapped.P
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        V = np.zeros(n_states)

        for _ in range(max_iter):
            delta = 0.0
            for s in range(n_states):
                q_values = []
                for a in range(n_actions):
                    # Using the Bellman update: Q(s,a) = sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V(s')]
                    q = sum(p * (r + gamma * V[s_]) for p, s_, r, _ in P[s][a])
                    q_values.append(q)
                v_new = max(q_values)
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break

        return V

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def state_to_features(self, state: int, grid_size: int) -> list[float]:
        """Encode state as [row/N, col/N, manhattan_dist/max_dist]."""
        N = grid_size
        row, col = divmod(state, N)
        goal_row, goal_col = N - 1, N - 1
        max_dist = 2 * (N - 1)
        dist = abs(row - goal_row) + abs(col - goal_col)
        return [row / N, col / N, dist / max_dist]

    def build_dataset(self, V: np.ndarray, grid_size: int):
        """Build (features, target) tensors from VI value table."""
        n_states = len(V)
        X = torch.tensor(
            [self.state_to_features(s, grid_size) for s in range(n_states)],
            dtype=torch.float32,
        )
        y = torch.tensor(V, dtype=torch.float32)
        return X, y

    def train_value_network(
        self, env, grid_size, gamma=0.99, epochs=1000, lr=1e-3, hidden_size=64, verbose=False
    ):
        """
        Returns:
            value_fn: callable (state: int) -> float
            model:    trained ValueMLP (for inspection / reuse)
        """
        # 1. Get VI targets
        V = self.value_iteration(env, gamma=gamma)

        # 2. Build dataset
        X, y = self.build_dataset(V, grid_size)

        # 3. Train MLP
        model = ValueMLP(hidden_size=hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  [ValueMLP] epoch {epoch + 1}/{epochs}  loss={loss.item():.6f}")

        model.eval()

        # 4. Return a simple callable
        def value_fn(state: int) -> float:
            feats = torch.tensor(self.state_to_features(state, grid_size), dtype=torch.float32)
            with torch.no_grad():
                return model(feats.unsqueeze(0)).item()

        return value_fn, model


class ValueFunctionOnly:
    """
    Replaces the rollout policy entirely. The value_fn can be the heuristic
    or the trained MLP from train_value_network().
    """

    def __init__(self, value_fn):
        self.value_fn = value_fn

    def __call__(self, node) -> float:
        return self.value_fn(node.state)
