# The Flax NNX API.
from flax import nnx  
from functools import partial

class CNN(nnx.Module):
    """A simple CNN model"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool,window_shape=(2, 2),strides=(2, 2),padding='SAME')
        self.linear1 = nnx.Linear(9984, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x
