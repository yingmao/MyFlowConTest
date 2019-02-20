from app.threadutils import RepeatedTimer
import subprocess


class TrialListener(object):

    def __init__(self, trial, interval=10):
        """Listen for new containers, manipulate a Trial's timer"""
        self._is_running = False
        self.trial = trial
        self.timer = RepeatedTimer(interval, self.listen)
        self.active_containers = []

    def start(self):
        # Keep the listener from being started twice
        if not self._is_running:
            self._is_running = True
            self.active_containers = get_active_containers()
            self.timer.start()

    def stop(self):
        self.timer.stop()
        self._is_running = False

    def listen(self):
        current_active = get_active_containers()
        if len(current_active) == 0:
            self.stop()
            self.trial.kill()
        else:
            # Reset the trial timer if there is a new container
            for container in current_active:
                if container not in self.active_containers:
                    self.trial.stop_backoff()
                    break


def get_active_containers():
    """Return the number of currently running containers"""
    out = subprocess.check_output(['docker', 'ps', '-q'])
    out = out.decode('ascii')
    out = out.split('\n')
    active_containers = []
    for line in out:
        if line != '':
            active_containers.append(line)
    return active_containers
