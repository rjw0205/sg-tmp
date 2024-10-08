import incl
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class InclLogger(Logger):
    @property
    def name(self):
        return "InclLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        incl.config.init(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step):
        incl.log(metrics, step=step)
