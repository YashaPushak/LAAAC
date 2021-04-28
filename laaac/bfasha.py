import logging
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.DEBUG)

class BFASHA:
    def __init__(self, grace_period=1, reduction_factor=5):
        self._grace_period = grace_period
        self._grace_period_index = 1
        self._reduction_factor = reduction_factor
        self._milestones = [(grace_period*reduction_factor, {}),
                            (grace_period, {}),
                            (0, {})]  # Dumby milestone
        self._released = defaultdict(list)
        self._times_suggested = defaultdict(set)
        self._times_reported = defaultdict(int)
        self._next_id = 0

    def suggest(self):
        id_to_run = None
        # First prioritize the configurations that have been released,
        # In order of those which appear to be the most promising
        logging.debug('Looking for a configuration that has been released.')
        for milestone_idx in range(1, len(self._milestones)):
            milestone_upper, recorded_upper = self._milestones[milestone_idx-1]
            milestone_lower, _ = self._milestones[milestone_idx]
            released_lower = self._released[milestone_lower]
            np.random.shuffle(released_lower)
            logging.debug(f'milestone_upper: {milestone_upper}; recorded_upper: {recorded_upper}; '
                          f'milestone_lower: {milestone_lower}; released_lower: {released_lower};')
            for config_id in released_lower:
                logging.debug(f'config_id: {config_id}; times_suggested: {self._times_suggested[config_id]};')
                if self._times_suggested[config_id] < milestone_upper:
                    id_to_run = config_id
                    break
            if id_to_run is not None:
                break
        # Next check to see if any reached a milestone and
        # can be released
        if id_to_run is None:
            logging.debug('Looking for a configuration to release')
            for milestone, recorded in self._milestones: 
                already_released = self._released[milestone]
                if len(already_released) < int(len(recorded)/self._reduction_factor):
                    # This milestone has enough recorded results to release one of them
                    for config_id in sorted(recorded, key=lambda k: recorded[k]):
                        if config_id in already_released:
                            continue
                        id_to_run = config_id
                        already_released.append(config_id)
                        break
                if id_to_run is not None:
                    break
        # None of the existing configurations are ready to be released. so
        # we start a new one instead.
        if id_to_run is None:
            id_to_run = self._next_id
            self._next_id += 1
            # Immediately release everything from the dumby milestone.
            self._released[0].append(id_to_run)
            # We also need to record a fake result for it
            self._milestones[-1][1][id_to_run] = np.inf
        # Track the fact that this id was suggested.
        self._times_suggested[id_to_run] += 1
        # Check to see if we need a bigger milestone soon
        if self._times_suggested[id_to_run] >= self._milestones[1][0]:
            self._milestones.insert(0, (self._milestones[0][0]*self._reduction_factor, {}))
            self._grace_period_index += 1
        # Return the id to run, as well as the number of times it was suggested
        # since each time an id is suggested we can get multiple reports and we
        # need to know to which one a report corresponds.
        return id_to_run, self._times_suggested[id_to_run]

    def report(self, config_id, time_suggested, loss):
        logging.debug(f'Catgoerical config_id: {config_id}; loss {loss}')
        self._times_reported[config_id].add(time_suggested)
        # The fidelity of a categorical arm is number of unique
        # numeric value configurations for which we have observed
        # a report of any level of fidelity.
        fidelity = len(self._times_reported[config_id])
        for milestone, recorded in self._milestones:
            if fidelity < milestone or config_id in recorded:
                continue
            else:
                recorded[config_id] = loss
                break
        logging.debug(f'Recorded data: {self._milestones}')

    def no_new_suggestions(self):
        """
        To be called when suggest suggests a new configuration, but there are
        no new ones to suggest and there is still budget left to spend.
        """
        # Track the fact that the configuration ID suggested has been rejected
        self._next_id -= 1
        _, recorded = self._milestones[self._grace_period_index]
        # Release everything that has ever been suggested from the previous
        # grace period.
        self._released[self._grace_period] = list(range(self._next_id))
        logging.debug(f'Released 0-{self._next_id-1} (if they were not already released.')
        # Update the grace period
        self._grace_period_index -= 1
        self._grace_period = self._milestones[self._grace_period_index][0]
        logging.debug(f'The new grace period is {self._grace_period}')
