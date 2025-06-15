from linalgzero.experiments.config import ZeroConfig


class ZeroTrainer:
    def __init__(self, config: ZeroConfig):
        """Base class for all trainers.

        The trainer is responsible for creating the data, model, loss, and optimiser.
        It also handles the training and evaluation of the model.

        Args:
            config (ZeroConfig): Config object.
        """
        self.config = config
