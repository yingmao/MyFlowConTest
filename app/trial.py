"""The Trial class, a singleton which manages given experiment.

The Trial.run() method can be thought of the top-level 'main' method. The reason it is encapsulated inside of
a Trial object rather than defined as a global function is as follows: the run method needs to be executed
repeatedly over a specific interval. So it needs to have an associated RepeatedTimer object. The Trial class keeps
the run method and the timer bound together. Further, it provides a place for the results of each iteration of algorithm 1
to accumulate.


"""

import sys
import os
import glob
import shutil
import zipfile
import logging
from app.listener import TrialListener

from app.algorithm import *
from app.threadutils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
fh = logging.FileHandler('FlowCon.log')
fh.setFormatter(formatter)
logger.addHandler(fh)


class Trial(object):
    """Manage one experimental run, called a Trial:
        repeatedly run the algorithm at a given time interval,
        save algorithm logs and resource monitor logs once no containers are active on the host machine

        Note that Trial will do its best to avoid duplicate
        experiment names by checking its name against logs left
        over by previous experiments. If it's name appears to be a duplicate, it will raise a ValueError

    TODO Ideal case: each container has one monitor
    """

    def __init__(self, alpha, name, interval, stats_interval, no_algo=False, no_update=False):
        """
        :param interval: the interval at which to run algorithm 1
        :param alpha: alpha for altorithm 1
        :param name: A name for the experiment Trial, passed as a command line arg.
        :param stats_interval: number of seconds between calls to docker stats: passed to ResourceMonitor
        """

        if glob.glob('./experiment_{}*.zip'.format(name)):
            raise ValueError("Logs for an experiment with name '{}' already exist, ".format(name) +
                             "please use unique experiment names")

        self.interval   = interval
        self.alpha      = alpha
        self.name       = name
        self.monitor    = ResourceMonitor(stats_interval)
        if no_algo or no_update:
            self.containers = ContainerList(no_update=True)
        else:
            self.containers = ContainerList(no_update=False)
        self.status     = None
        self.interval = interval
        self.backoff_interval = interval  # for the exponential backoff
        self.stats_interval = stats_interval
        self.iter_num = 0
        self.start_time = time.time()
        self._fn = 'watching_completing.csv'  # TODO put name here
        self.no_algo = no_algo
        self.no_update = no_update
        self.listener = TrialListener(self)
        self.timer = RepeatedTimer(self.interval, self.run, self.containers, self.monitor)
        self.timer.start()
        self._make_logfile()

        logger.info("Created Trial object with parameters name = {}, alpha = {}, interval = {}".format(name, alpha, interval))

    def _make_logfile(self):
        """Create a logfile for the Trial."""
        with open(self._fn, 'w') as f:
            f.write('iter, num_watching, num_completing, total\n')

    def backoff(self):
        self.backoff_interval *= 2
        logger.info("Backing off algo interval to {}".format(self.backoff_interval))
        self.timer.stop()
        self.timer = RepeatedTimer(self.backoff_interval, self.run, self.containers, self.monitor)
        self.timer.start()
        self.listener.start()

    def stop_backoff(self):
        self.timer.stop()
        self.backoff_interval = self.interval
        logger.info("Resetting algo interval to {}".format(self.interval))
        self.timer = RepeatedTimer(self.interval, self.run, self.containers, self.monitor)
        self.timer.start()
        self.listener.stop()

    def run(self, containers, monitor):
        """The main procedure of an Trial

        :param containers: the ContainerList
        :param monitor: the DockerMonitor
        :return: None

        This gets executed by self.timer every self.interval seconds

        In pseudocode:

            check for new containers and terminated containers:
                update ContainerList and save logs accordingly
            run algorithm 1 over the ContainerList
            append the results of algorithm1 to self.status
            write the cardinality of containers in (watching, completing, and total) to the appropriate log
            update ContainerList
            if ContainerList is empty:
                save all logs
                zip all logs
                exit
        """
        logger.debug("Executing Trial.run()")
        containers.reconcile(experiment_name=self.name)
        if not self.no_algo and len(containers) > 0:
            status = algo_1(containers, monitor, alpha=self.alpha, interval=self.interval, no_update=self.no_update)

            if self.containers.all_completing:
                self.backoff()

            delta_t = round(time.time() - self.start_time, 2)
            status.insert(1, 'delta_t', delta_t)
            status.insert(2, 'iter', self.iter_num)
            self.iter_num += 1

            if self.status is None:
                self.status = status
            else:
                self.status = pd.concat([self.status, status])

            print(status)

            with open(self._fn, 'a') as f:
                f.write('{}, {}, {}, {}\n'.format(self.iter_num, self.containers.num_watching,
                                                  self.containers.num_completing, len(self.containers)))

        containers.reconcile(experiment_name=self.name)

        if len(containers) == 0:
            self.kill()


    def to_csv(self):
        logger.debug("Writing Trial records to CSV")
        if not self.no_algo:
            self.status.to_csv('{}_algo_1_iters.csv'.format(self.name), index=False)
        self.monitor.to_csv(self.name)

    def kill(self):
        logger.debug('Killing Trial Instance')
        self.containers.killall(self.name)
        self.to_csv()
        self.zip_logs()
        self.timer.stop()
        self.monitor.kill()
        sys.exit(0)

    def zip_logs(self):
        """Move all log files to a separate directory, zip them and delete the raw log files"""
        new_dir = "./{}".format(self.name)
        logger.debug("Zipping Trial records")
        os.makedirs(new_dir)
        for file in glob.glob("{}*".format(self.name)):
            shutil.move(file, new_dir)

        with zipfile.ZipFile('{}_logs.zip'.format(self.name), 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(new_dir):
                for file in files:
                    zf.write(os.path.join(root, file))

        shutil.rmtree(new_dir)
