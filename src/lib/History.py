from __future__ import absolute_import, print_function


class History(object):
    """ history object to record statistics of training """
    def __init__(self, loss_columns):
        self.lossCols = loss_columns
        self.history = {"epoch": []}
        for lc in loss_columns:
            self.history[lc] = []

    def append(self, epoch, losses):
        self.history["epoch"].append(epoch)
        for i, loss in enumerate(losses):
            self.history[self.lossCols[i]].append(loss)

    def getHistory(self):
        return self.history