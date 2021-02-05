from collections import deque

from agents.utils import print_warning


class MemoryBuffer:

    def __init__(self, values, buffer_size):

        """
        Stores the information that an agent receives temporarily before it can be stored in the replay buffer.
        :param values: Names of the datapieces that will be stored, List of str
        :param buffer_size: Maximum buffer size, int
        """

        self.buffer_size = buffer_size
        self.values = values
        self.data = {value: deque() for value in self.values}
        self.size = 0

        return

    def store(self, **kwargs):

        """
        Stores step of information.
        :param kwargs: Dictionary of values that is consistent with the values in self.values.
        :return:
        """

        if self.size >= self.buffer_size:
            [value.popleft() for key, value in self.data.items()]

            print_warning('Warning: Memory buffer has reached its maximum size and will now drop the least recent '
                              'information. Please adjust the buffer size or evaluate your code for bugs if this '
                              'behavior is not intended.')
        else:
            self.size += 1

        [self.data[key].append(value) for key, value in kwargs.items()]

        return

    def retrieve(self, steps):

        print(steps)
        print(self.size)

        if steps > self.size:
            print_warning(f'Warning: Requested steps ({steps}) is bigger than current buffer size ({self.size}). '
                          f'All available information will be returned.')
            steps = self.size
        ans = {value: [] for value in self.values}
        [[ans[key].append(value.popleft()) for key, value in self.data.items()] for _ in range(steps)]
        self.size -= steps

        return ans.values()


'''
print('Testing Memorybuffer')

test_rounds = 10

mb = MemoryBuffer(values=['a', 'b', 'c'], buffer_size=5)


for i in range(test_rounds):
    data = {'a': np.array([1*i, 1]),
            'b': np.array([2*i, 1]),
            'c': np.array([3*i,1]),}

    mb.store(a=data['a'], b=data['b'], c=data['c'])

    print(mb.data)

    a, b, c = mb.retrieve(steps=2)

    print(a)
    print(b)
    print(c)

print(mb.size)
print(mb.data)
print('Done')

exit()

'''