import random


class ReplayBuffer:
    def __init__(self, buffer_size=512):
        self.buffer_size = buffer_size
        #
        self.buffer = list()

    def insert(self, *args):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(args)

    def get_batch(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def reset(self):
        self.buffer.clear()