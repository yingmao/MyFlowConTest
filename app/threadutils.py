"""Classes to handle threading
"""

from threading import Timer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
fh = logging.FileHandler('FlowCon.log')
fh.setFormatter(formatter)
logger.addHandler(fh)


class RepeatedTimer(object):
    """
    Executes a function with arbitrary arguments every `interval` seconds

    from: https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds
    """
    def __init__(self, interval=30, function=None, *args, **kwargs):
        """
        :param interval: the number of seconds to wait before calling `function` again
        :param function: the function to execute every `interval` seconds
        :param args: positional arguments to `function`
        :param kwargs: keyword arguments to `function`
        """
        logger.debug("Initializing RepeatedTimer instance with function: {}, interval: {}".format(function.__name__,
                                                                                                  interval))
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.is_running = False
        self.start()

    def _run(self):
        logger.debug("Executing RepeatedTimer._run with function: {}".format(self.function.__name__))
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        logger.debug("Starting RepeatedTimer._timer")
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        logger.debug("Stopping RepeatedTimer._Timer object")
        self._timer.cancel()
        self.is_running = False
