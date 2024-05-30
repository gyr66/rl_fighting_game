import os
from stable_baselines3.common.callbacks import BaseCallback
from opponent_pool import OpponentPool

class SaveCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``save_freq`` steps)

    :param save_freq: (int)
    :param save_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """
    def __init__(self, save_freq: int, save_dir: str, verbose=1):
        super(SaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"Saving model_{self.num_timesteps} checkpoint to {self.save_dir}", flush=True)
            model_name = f"model_{self.num_timesteps}"
            self.model.save(os.path.join(self.save_dir, model_name))
        return True