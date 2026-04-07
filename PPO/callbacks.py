from stable_baselines3.common.callbacks import BaseCallback


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_episode = []
        self.entropy_losses = []
        self.approx_kls = []
        self.training_timesteps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_at_episode.append(self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        try:
            logger = self.model.logger.name_to_value
            self.entropy_losses.append(logger.get("train/entropy_loss", float("nan")))
            self.approx_kls.append(logger.get("train/approx_kl", float("nan")))
            self.training_timesteps.append(self.num_timesteps)
        except AttributeError:
            pass