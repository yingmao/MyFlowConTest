"""Classes to manage Docker containers


        Make sure that all jobs are configured to:
            write consistently formatted log statements
            flush stdout after every log statement

        Have a directory called /docker_data where all of your jobs are located

        Have all of your required docker images installed

    TODO we need to tune alpha and time interval for each model
"""
import subprocess
from subprocess import DEVNULL
from multiprocessing import cpu_count
import re
import time
import warnings
import logging

import pandas as pd
import numpy as np


from app.threadutils import RepeatedTimer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
fh = logging.FileHandler('FlowCon.log')
fh.setFormatter(formatter)
logger.addHandler(fh)

class ContainerWrapper(object):
    """A python interface to docker containers running ML jobs

    Allows us to monitor the state of evaluation functions and update resource limits.
    """

    def __init__(self, id=None, create=False, image=None, wd=None, script=None, njobs=1):
        """
        :param id: Container ID: if create=True then this has no effect
        :param create: if True, the ContainerWrapper will create a container based on `image`, `wd`, and `script`
        :param image: see `create`
        :param wd: see `create`
        :param script: see `create`
        :param njobs: number of ML jobs running within the container. Currently only supports 1.
        """
        self.id = id
        if create:
            self._run(image, wd, script)
        self.mem_lim        = None
        self.cpu_lim        = None
        self.njobs          = njobs
        self.watching       = None   # TODO: In my option, these properties are pretty sloppy OO.
        self.completing     = None   # They are essentially using a ContainerWrapper object to store data for logic
        self.frozen         = False  # external to the container object leading to class bloat
        self._creation_time = time.time()
        if njobs != 1:
            raise NotImplementedError('Currently only supports one job')

    def _run(self, image, wd, script):
        """ Create a container from an image. Currently this only works on the mtynes_docker_kube CloudLab image.
    
        :param image: the image to use to create a container
        :param wd: the working directory of the created container
        :param script: the python script to run on the container
        :return: None
        """
        command_string = "docker run -d -v /docker_data:/root/docker_data " + \
                         "-w {} {} python {}".format(wd, image, script)
        self.id = subprocess.check_output(command_string.split()).decode('ascii')[:-1]

    @property
    def loss_table(self):
        """Parse the container logs and return a pd.DataFrame of the loss function over the lifetime of the container"""

        if self.njobs == 1:
            logs = subprocess.check_output(['docker', 'logs', self.id])
            logs = logs.split(b"\n")
            loss = []
            timestamp = []
            for line in logs[:-1]:
                try:
                    l = float(re.search(b'Loss: ([0-9.]+)', line).group(1))
                    t = float(re.search(b'Time: ([0-9.]+)', line).group(1))
                except AttributeError:  # If re.search returns NoneType, which has no attribute 'group'
                    continue
                else:
                    loss.append(l)
                    timestamp.append(t)

            history = pd.DataFrame({'loss': loss, 'time': timestamp})

        else:
            raise NotImplementedError("This should never happen: currently only supports one job")
            # When more than one job is supported, this method will have to change

        return history

    @property
    def cpu_lim(self):
        """CPU limit placed on container where the unit is the number of cpus

        Setting cpu_lim causes an instance to run `docker update self.id --cpus limit`
        """
        return self._cpu_lim

    @cpu_lim.setter
    def cpu_lim(self, limit):
        if limit is not None:
            logger.info("Setting container {} cpu limit to {}".format(self.id, limit))
            response = subprocess.check_output(['docker', 'update', '--cpus', str(limit), self.id])
            logger.info("Docker response: {}".format(response))
        self._cpu_lim = limit

    @property
    def mem_lim(self):
        """Memory limit placed on container

        Setting mem_lim causes an instance to run `docker update self.id --memory limit`
        """
        return self._mem_limit

    @mem_lim.setter
    def mem_lim(self, limit):
        if limit is not None:
            response = subprocess.check_output(['docker', 'update', '--memory', str(limit), self.id])
            logger.info("Setting container {} memory limit to {}".format(self.id, limit))
            logger.info("Docker response: {}".format(response))
        self._mem_limit = limit

    @property
    def age(self):
        return time.time() - self._creation_time

    def growth_tuple(self, monitor, interval, threshold=0):
        """Compute the growth efficiency for a container; return loss, progress, and growth

        GROWTH WILL BE SET TO 0 if no history yet, if this happens twice it will be marked as completing...

        :param container: a ContainerWrapper object
        :param monitor: a DockerMonitor object
        :param interval: the time interval over which to compute growth efficiency in seconds
        :param threshold: currently unused, resource must be above a certain threshold
        :return: a triple: (E_i: loss at of the container over the interval,
                            progress_score of the container over the interval,
                            growth efficiency of the container over the interval)
        """
        logger.info("Generating growth tuple for container {} with interval {}".format(self.id, interval))

        E_i, progress_score = self._loss_and_progress(interval)

        if progress_score is None:
            logger.info("Returning None for growth score")
            return E_i, None, None

        resources = monitor.history
        resources = resources[resources.container_id == self.id]
        resources = resources[resources.time >= (time.time() - interval)]

        if resources.shape[0] == 0:
            # then we dont have any resource history for this container yet, so it cant have grown efficiently.

            warn_str = "No resources history in this interval  for container: {}, returning growth of 0".format(self.id)
            warnings.warn(warn_str, RuntimeWarning)
            logger.warning(warn_str)
            return E_i, None, None

        resources['cpu_pct'] = resources.cpu_pct.str.rstrip('%').astype(float)
        resources['cpu_norm'] = resources.cpu_pct / cpu_count() / 100
        resources['mem_norm'] = resources.mem_pct.str.rstrip('%').astype(float) / 100
        cpu_mean = resources.cpu_norm.mean()

        if cpu_mean < threshold:
            raise NotImplementedError("Got CPU mean of {}, which is <= threshold of {}".format(cpu_mean, threshold))
        else:
            growth = progress_score / cpu_mean
            logger.info("Returning loss = {} progress = {} growth = {} for {}".format(E_i, progress_score, growth, self.id))
            return E_i, progress_score, growth

    def save_logs(self, experiment_name):
        """Save loss function table to csv

        :param experiment_name: the name of the controlling Trial object
        :return: None
        """
        table = self.loss_table
        logger.info("Saving logs for container {}".format(self.id))
        table.to_csv("{}_{}.csv".format(experiment_name, self.id), index=False)

    def kill(self):
        """Kill the container controlled by self"""
        subprocess.run(['docker', 'container', 'kill', self.id], stdout=DEVNULL)

    def _loss_and_progress(self, interval):
        """Compute the loss and progress score over the `interval` for use in Algorithm 1

        :param: interval: number of seconds defining a time interval
        :return: (loss over interval, progress score over interval)
        """


        logger.info('Computing mean loss over intervals i and i-1 progress scores')
        loss_history = self.loss_table
        loss_history['loss'] = loss_history.loss / loss_history.loss.max()  # normalize loss
        now = time.time()

        # See writeup of Algorithm 1 in paper to disambiguate notational choices here
        loss_over_this_interval = loss_history.loss[loss_history.time >= now - interval]
        loss_over_previous_interval = loss_history.loss[(now - 2 * interval <= loss_history.time) & (loss_history.time <= now - interval)]

        E_i = loss_over_this_interval.mean()
        E_i_minus_1 = loss_over_previous_interval.mean()

        if len(loss_over_previous_interval) == 0:
            logger.info("No loss over previous interval, returning None for progress score")
            return E_i, None
        else:
            progress = abs(E_i - E_i_minus_1) / interval
            return E_i, progress


class ContainerList(object):
    """A list-like object for storing ContainerWrappers"""

    def __init__(self, no_update=False, *args):
        """Create self from a comma-separated list of ContainerWrappers
        :param *args: ContainerWrapper objects to store in instance
        """

        logger.info("Initializing ContainerList")
        self.no_update = no_update
        self.containers = []
        self.add(*args)

    def add(self, *args):
        """Add more containers to self

        :param *args: ContainerWrapper objects to store in instance
        :return: None
        """
        for arg in args:
            if not isinstance(arg, ContainerWrapper):
                logger.error("ContainerList passed non-ContainerWrapper object")
                raise ValueError("ContainerList can only take ContainerWrapper objects, got {}".format(type(arg)))

        # logger.info("Adding {} containers to ContainerList".format(len(args)))
        self.containers.extend(list(args))

    def reconcile(self, experiment_name):
        """Reconcile the state of the container list with the state of currently active containers

        Add any newly created containers running on the machine to self, and remove those that have terminated

        :param experiment_name: the name of the controlling Trial instance
        :return: None
        """

        logger.info('Reconciling ContainerList with docker ps')

        active_containers = subprocess.check_output(['docker', 'ps', '-q']).decode('ascii').split('\n')[:-1]

        for c_id in active_containers:
            if c_id not in self.ids:
                c = ContainerWrapper(id=c_id)
                logger.info('Adding {} to ContainerList'.format(c_id))
                self.add(c)

        for c in self:
            if c.id not in active_containers:
                logger.info('Removing {} from ContainerList'.format(c.id))
                c.save_logs(experiment_name=experiment_name)
                self.containers.remove(c)
            if c.cpu_lim is None:
                new_lim = cpu_count()
                if not self.no_update:
                    logger.info("Container {} has limit = None, updating...".format(c.id))
                    c.cpu_lim = new_lim  # TODO this is a rather strange place for this to happen

    def __iter__(self):
        for container in self.containers:
            yield container

    def __len__(self):
        return len(self.containers)

    def killall(self, experiment_name, save_logs=True):
        """Kill all ContainerWrappers in self"""
        for container in self:
            if save_logs:
                container.save_logs(experiment_name=experiment_name)
            container.kill()

    @property
    def all_completing(self):
        """Check if all containers in self have been marked as 'completing' by the algorithm
        :return: bool
        """
        for container in self:
            if not container.completing:
                return False
        return True

    @property
    def num_completing(self):
        """Return the number of containers in self that the algorithm has marked as completing"""
        return sum(1 for c in self if c.completing)

    @property
    def num_watching(self):
        """Return the number of containers in self that the algorithm has marked as completing"""
        return sum(1 for c in self if c.watching)

    @property
    def ids(self):
        """Return a list of container IDs corresponding to the containers stored in self"""
        return [c.id for c in self]


class ResourceMonitor(object):
    """An object that maintains a table of docker resource usage statistics

    Meant to be used as a singleton.

    Runs `docker stats --no-stream` every n seconds using a RepeatedTimer object, accumulating results into a DataFrame
    """

    def __init__(self, update_interval=10):
        """
        :param update_interval: how frequently, in seconds, to update docker stats table
        """
        logger.info('Initializing ResourceMonitor with update interval = {}'.format(update_interval))
        self.history = self._check_stats()
        self._update_interval = update_interval
        self._timer = RepeatedTimer(interval=self._update_interval, function=self._update)
        self._timer.start()
    def _check_stats(self):
        """Run `docker stats --no-stream` and parse into pd.DataFrame

        TODO the columns printed vary with docker versions... standardize this somehow.
        TODO note: had to install docker version 17 and anaconda on chameleon for this to work
        """

        logger.debug('ResourceMonitor: checking stats')
        columns = ['container_id', 'cpu_pct', 'mem_use', 'mem_max',
                   'mem_pct', 'net_in', 'net_out', 'block_in', 'block_out', 'pids']

        records = subprocess.check_output(['docker', 'stats', '--no-stream']).decode('ascii')
        records = records.split('\n')[1:-1]  # exclude headers and trailing empty string
        records = [re.split('[ /]+', record) for record in records]

        stats = pd.DataFrame.from_records(records, columns=columns)
        stats['time'] = time.time()
        logger.debug('ResourceMonitor: done checking stats')
        return stats

    def _update(self):
        """Run self._check_stats() and concatenate to self.history"""
        self.history = pd.concat([self.history, self._check_stats()], ignore_index=True)

    def kill(self):
        """Kill the RepeatedTimer thread"""
        self._timer.stop()

    def to_csv(self, experiment_name):
        """Save self.history to a csv

        :param experiment_name: the name of the controlling Trial instance
        :return: None
        """
        logger.info("Writing ResourceMonitor table to csv")
        self.history.to_csv("{}_docker_stats.csv".format(experiment_name), index=False)
