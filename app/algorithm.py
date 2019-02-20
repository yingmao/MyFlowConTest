"""This module implements algorithm 1 from the paper
"""
import logging

from app.dockerutils import *
import multiprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
fh = logging.FileHandler('FlowCon.log')
fh.setFormatter(formatter)
logger.addHandler(fh)


def algo_1(containers, monitor, alpha=0.05, interval=30, no_update=False):
    """Run algorithm1 over a ContainerList
    :param containers: the ContainerList for the session
    :param monitor: the DockerMonitor for the session
    :param alpha: decision threshold for growth efficiency
    :param interval: time interval over which to run the algorithm
    :return: a pandas DF of the status of all monitored containers after the run of the algorithm

    TODO refactor such that interval and alpha can vary independently for each container
    """

    logging.info("Running algorithm 1 with parameters alpha = {}, interval = {}".format(alpha, interval))

    # accumulators for pandas DF
    growth = [None] * len(containers)
    loss = [None] * len(containers)
    progress = [None] * len(containers)
    ages = [None] * len(containers)
    ignore = [False] * len(containers)

    for i, c in enumerate(containers):
        l, P, G = c.growth_tuple(monitor, interval)
        growth[i] = G
        loss[i] = l
        progress[i] = P
        ages[i] = c.age

        if G is None:
            logger.info('Ignoring container {}'.format(c.id))
            ignore[i] = True
            continue

        # check conditions
        if G < alpha and not c.watching and not c.completing:
            logger.info("Marking {} as watching".format(c.id))
            c.watching = True
            c.completing = False
        elif G < alpha and c.watching and not c.completing:
            logger.info("Marking {} as completing".format(c.id))
            c.watching = False
            c.completing = True
        elif G >= alpha:
            logger.info("Marking {} as neither watching nor completing".format(c.id))
            c.completing = False
            c.watching = False

    if containers.all_completing and not no_update:
        try:
            new_lim = 1.5 * 1/len(containers)
        except ZeroDivisionError:
            logger.warning("Containers finished while algorithm running", exc_info=True)
        else:
            new_lim = min(new_lim, 1)
            new_lim = new_lim * multiprocessing.cpu_count()
            for c in containers:
                logger.info('freezing container {} limit to 1/n'.format(c.id))
                c.frozen = True
                c.cpu_lim = new_lim

    elif containers.num_watching + containers.num_completing != len(containers):
            # Apply resource limits from lines 16-22 of the algorithm as written in the paper
            growth_sum = sum(filter(None, growth))
            logger.info("Value for growth sum: {:.3f}".format(growth_sum))
            for i, c in enumerate(containers):
                current_normalized_lim = c.cpu_lim / multiprocessing.cpu_count()

                if c.completing:
                    growth = growth[i] if growth[i] is not None else 0  # TODO it seems like some containers can have None
                    multiplier = (1 - (growth / (growth_sum + 1e-10)))  # for growth when they are completing...
                elif c.watching or ignore[i]:
                    continue
                else:
                    multiplier = (1 + (growth / growth_sum))

        #        multiplier = max(0.5, multiplier)
                new_lim = current_normalized_lim * multiplier

                try:
                    new_lim = max(new_lim, 1/10*len(containers))
                except ZeroDivisionError:
                    logger.warning("Containers finished while algorithm running", exc_info=True)
                else:
                    new_lim = min(new_lim, 1)
                    if c.frozen:
                        logger.info('container {} is frozen at 1/n'.format(c.id))
                        new_lim = 1 / len(containers)
                    if not no_update:
                        logger.info("Updating container {} with\tgrowth={}\tmultiplier={}".format(c.id, growth[i], multiplier))
                        new_lim_un_normalized = new_lim * multiprocessing.cpu_count()
                        c.cpu_lim = round(new_lim_un_normalized, 2)
    else:
        # keep frozen containers frozen even if the previous block doesnt get hit
        for c in containers:
            if c.frozen and not no_update:
                new_lim = 1 / len(containers) * multiprocessing.cpu_count()
                c.cpu_lim = new_lim
            if c.watching and not no_update:
                c.cpu_lim = 1.5 / len(containers) * multiprocessing.cpu_count()



    now = time.time()
    limits = [c.cpu_lim for c in containers]
    W = [c.watching for c in containers]
    C = [c.completing for c in containers]
    ids = [c.id for c in containers]

    status = pd.DataFrame(dict(
        time=now,
        c_id=ids,
        age=ages,
        ignore=ignore,
        loss=loss,
        progress=progress,
        growth=growth,
        limit=limits,
        watching=W,
        completing=C,
    ))
    status = status[['time', 'age', 'ignore', 'c_id', 'loss', 'progress', 'growth', 'limit', 'watching', 'completing']]

    normalized_limit = status['limit'] / multiprocessing.cpu_count()
    status.insert(8, 'limit_norm', normalized_limit)
    return status
