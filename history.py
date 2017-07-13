import random


class ReplayBuffer:
    def __init__(self, buffer_len=512):
        self._buffer_len = buffer_len
        self._buffer = list()

    def insert(self, *args):
        if len(self._buffer) > self._buffer_len:
            self._buffer.pop(0)
        self._buffer.append(args)

    def get_batch(self, n_batch, n_lost):
        # forget random items
        for _ in range(n_lost):
            self._buffer.pop(random.randrange(self.len()))
        # gather a mini batch
        batch = random.sample(self._buffer, n_batch)
        return zip(*batch)

    def reset(self):
        self._buffer.clear()

    def len(self):
        return len(self._buffer)

    def is_full(self):
        return len(self._buffer) == self._buffer_len
