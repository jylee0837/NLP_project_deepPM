#!/usr/bin/env python

import os
import sys

HOME = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

class Experiment(object):
    def __init__(self, name, time):
        # type: (str, str, str, List[str], List[str]) -> None
        self.name = name
        self.time = time

    def experiment_root_path(self):
        # type: () -> str
        return os.path.join(HOME, 'saved', self.name, self.time)

    def checkpoint_file_dir(self):
        # type: () -> str
        return os.path.join(self.experiment_root_path(), 'checkpoints')

    def checkpoint_file_name(self, run_time):
        # type: (float) -> str
        return os.path.join(self.checkpoint_file_dir(), '{}.mdl'.format(run_time))

