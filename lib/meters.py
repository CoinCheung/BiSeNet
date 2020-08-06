
import time
import datetime

class TimeMeter(object):

    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.st = time.time()
        self.global_st = self.st
        self.curr = self.st

    def update(self):
        self.iter += 1

    def get(self):
        self.curr = time.time()
        interv = self.curr - self.st
        global_interv = self.curr - self.global_st
        eta = int((self.max_iter-self.iter) * (global_interv / (self.iter+1)))
        eta = str(datetime.timedelta(seconds=eta))
        self.st = self.curr
        return interv, eta


class AvgMeter(object):

    def __init__(self, name):
        self.name = name
        self.seq = []
        self.global_seq = []

    def update(self, val):
        self.seq.append(val)
        self.global_seq.append(val)

    def get(self):
        avg = sum(self.seq) / len(self.seq)
        global_avg = sum(self.global_seq) / len(self.global_seq)
        self.seq = []
        return avg, global_avg

