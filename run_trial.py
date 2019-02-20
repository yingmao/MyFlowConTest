"""The main point of entry for this program"""

import argparse
import logging

from app.trial import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
fh = logging.FileHandler('FlowCon.log')
fh.setFormatter(formatter)
logger.addHandler(fh)


def run_job_list(job_list):
    """"""
    jobs = pd.read_csv(job_list)
    stop = jobs.seconds.max()

    for i in range(stop+1):
        jobs_i = jobs[jobs.seconds == i]
        # print('i:', i, '\njobs_i:\n', jobs_i)
        if jobs_i.shape[0] > 0:
            for job in jobs_i.images:
                subprocess.Popen(['docker', 'run', job], stdout=DEVNULL)
                logger.info('Launching container with `docker run {}`'.format(job))
        time.sleep(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('joblist', help='A csv of jobs to run')
    parser.add_argument('-i', '--interval', type=int, default=30,
                        help='The interval at which to run algorithm 1')
    parser.add_argument('-a', '--alpha', type=float, default=0.03,
                        help='Rate at which to change resource allocation')
    parser.add_argument("--docker_stats_interval", type=int, default=30,
                        help="Number of seconds between calls to `docker stats`")
    control = parser.add_mutually_exclusive_group()
    control.add_argument('--no_update', action='store_true',
                         help='Run the algorithm but do not update any container limits')
    control.add_argument('--no_algo', action='store_true',
                         help='Do not run the algorithm')

    args = parser.parse_args()
    for arg, val in vars(args).items():
        logger.info("Argument {}: {}".format(arg, val))

    #TODO stop if docker containers are already running
    session_name = "no_algo" if args.no_algo \
                   else "no_update" if args.no_update \
                   else "a{}_i{}".format(args.alpha, args.interval)

    logger.info(
        "Running trial with arguments a = {}, i = {}, name = {}".format(args.alpha, args.interval, session_name))
    session = Trial(interval=args.interval, name=session_name, alpha=args.alpha, no_algo=args.no_algo,
                    no_update=args.no_update, stats_interval=args.docker_stats_interval)
    run_job_list(args.joblist)
