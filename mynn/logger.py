from utils import OB


class Logger:
    def __init__(self, log_fn=OB.log):
        self.log_fn = log_fn

    def __call__(self, content_str):
        self.log_fn(content_str)

    def register_log_fn(self, log_fn):
        self.log_fn = log_fn

    def default_log_fn(self):
        self.log_fn = OB.log


NNLogger = Logger()
