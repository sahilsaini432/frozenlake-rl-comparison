"""
callbacks.py: for logging per episode and per rollout training stats
"""
from stable_baselines3.common.callbacks import BaseCallback

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super().__init__(verbose)
        self.ep_rewards = []
        self.ep_lens = []
        self.ts_e = []
        self.en_loss = []
        self.approx_kls = []
        self.train_ts = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lens.append(info["episode"]["l"])
                self.ts_e.append(self.num_timesteps)
        return True

    def _on_rollout_end(self):
        try:
            # only after at least one train() call
            lg = self.model.logger.name_to_value
            self.en_loss.append(lg.get("train/entropy_loss", float("nan")))
            self.approx_kls.append(lg.get("train/approx_kl", float("nan")))
            self.train_ts.append(self.num_timesteps)
        except AttributeError:
            pass