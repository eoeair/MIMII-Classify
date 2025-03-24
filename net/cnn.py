# The Flax NNX API.
from flax import nnx  
from functools import partial

class CNN(nnx.Module):
    """A simple CNN model"""

    def __init__(self, rngs: nnx.Rngs, return_latent=True, num_classes=3):
        self.return_latent = return_latent
        
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool,window_shape=(2, 2),strides=(2, 2),padding='SAME')
        self.map = nnx.Linear(9984, 256, rngs=rngs)
        self.embed = nnx.Linear(256, 2, rngs=rngs)
        self.logits = nnx.Linear(2, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.map(x))
        latent = self.embed(x)
        logits = self.logits(latent)
        if self.return_latent:
            return latent, logits
        else:
            return logits