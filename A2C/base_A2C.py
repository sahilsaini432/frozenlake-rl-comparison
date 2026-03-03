from stable_baselines3 import A2C
import torch as th

# Ths files shows the structure to create a agent using A2C as base


class ModA2C(A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        # set training mode for policy
        self.policy.set_training_mode(True)

        # train on the collected batch of data
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions

            # evaluate actions based on the current policy given the observations and actions from the rollout buffer
            # returns - estimated value, log likelihood of taking those actions and entropy of the action distribution.
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

            # relative improvement over baseline
            # Advantage = (Reward received + Discounted value of next state) - Value of current state
            advantages = rollout_data.advantages

            # Normalize advantages for better training stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Increases probability of actions with positive advantages, decreases probability of actions with negative advantages.
            # policy loss with additional penalty
            # Responsible for updating actor for action selection.
            policy_loss = -(advantages * log_prob).mean()

            # value loss
            # updates critic (state value estimation)
            value_loss = th.nn.functional.mse_loss(rollout_data.returns, values)

            # High entropy = more random/exploratory actions
            # Low entropy = more deterministic/exploitative actions
            # Negative sign in loss = reward higher entropy
            entropy_loss = -th.mean(entropy)

            # Combined loss with weights
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # Backpropagation
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Limits the magnitude of gradients during backpropagation to prevent exploding gradients
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # rollout collection with reward processing
        rollout_buffer.reset()
        callback.on_rollout_start()

        for step in range(n_rollout_steps):
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            # reward modification (Think of some more ways to do reward analysis and shaping)
            rewards = self.reward_processing(rewards, new_obs, dones)

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)

            self._last_obs = new_obs
            #  if dones is True, the next step is the start of a new episode.
            self._last_episode_starts = dones

        return True

    def reward_processing(self, rewards, obs, dones):
        # reward shaping logic
        return rewards * 2.0 if rewards > 0 else rewards - 0.5

    # Prediction logic with action selection strategy
    def select_action(self, observation, state=None, episode_start=None, deterministic=False):
        # prediction logic
        self.policy.set_training_mode(False)

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.policy.get_distribution(observation).get_actions(deterministic=deterministic)

            # Apply action selection strategy
            if not deterministic:
                actions = self.action_selection(actions, observation)

        actions = actions.cpu().numpy()

        return actions.item()

    def action_selection(self, actions, obs):
        # Exploration strategies during action selection
        return actions
