"""
ModPPO:SB3 PPO subclass for experimentation and customization on FrozenLake-v1

Keeps the PPO training loop close to SB3 and logs metrics
"""

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
 
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
 
 
class ModPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
 
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
 
        ent_losses = []
        pg_losses = []
        val_losses = []
        clip_fracs = []
        keep_training = True
 
        for epoch in range(self.n_epochs):
            kl_divs = []
            for rd in self.rollout_buffer.get(self.batch_size):
                actions = rd.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
 
                values, log_prob, entropy = self.policy.evaluate_actions(rd.observations, actions)
                values = values.flatten()
 
                advantages = rd.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio  = th.exp(log_prob - rd.old_log_prob)
                pl1 = advantages * ratio
                pl2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = -th.min(pl1, pl2).mean()
                pg_losses.append(pg_loss.item())
                clip_frac = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fracs.append(clip_frac)
                if clip_range_vf is None:
                    vpred = values
                else:
                    vpred = rd.old_values + th.clamp(values - rd.old_values, -clip_range_vf, clip_range_vf)
                val_loss = F.mse_loss(rd.returns, vpred)
                val_losses.append(val_loss.item())
 
                # Negative because add entropy for exploration
                if entropy is None:
                    ent_loss = -th.mean(-log_prob)
                else:
                    ent_loss = -th.mean(entropy)
                ent_losses.append(ent_loss.item())
                loss = pg_loss + self.ent_coef * ent_loss + self.vf_coef * val_loss
 
                with th.no_grad():
                    log_ratio = log_prob - rd.old_log_prob
                    kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    kl_divs.append(kl_div)
 
                if self.target_kl is not None and kl_div > 1.5 * self.target_kl:
                    keep_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {kl_div:.2f}")
                    break
 
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
 
            self._n_updates += 1
            if not keep_training:
                break
 
        exp_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(ent_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(val_losses))
        self.logger.record("train/approx_kl", np.mean(kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fracs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", exp_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
 
    def select_action(self, observation, deterministic = False):
        action, state = self.predict(observation, deterministic = deterministic)
        return int(action)