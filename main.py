# The Flax NNX API.
from flax import nnx  
from functools import partial

import optax

# JAX NumPy
import jax.numpy as jnp  
import jax_dataloader as jdl

from tqdm import tqdm

from net import CNN
from feeder import *

def loss_fn(model: CNN, batch):
  logits = model(batch['data'])
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
  return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

@nnx.jit
def pred_step(model: CNN, batch):
  logits = model(batch['data'])
  return logits.argmax(axis=1)

if __name__ == '__main__':
    
    # Set the global seed to 0 for reproducibility
    jdl.manual_seed(0) 
    train_loader = jdl.DataLoader(
      Feeder_snr('data/train_data.npy','data/train_label.npy'), # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
      backend='jax', # Use 'jax' backend for loading data
      batch_size=256, # Batch size 
      shuffle=True, # Shuffle the dataloader every iteration or not
      drop_last=False, # Drop the last batch or not
  )

    test_loader = jdl.DataLoader(
      Feeder_snr('data/test_data.npy','data/test_label.npy'),
      backend='jax', 
      batch_size=256,
      shuffle=True, 
      drop_last=False,
  )
    
    # Instantiate the model.
    model = CNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-4,b1=0.9))
    metrics = nnx.MultiMetric(
      accuracy=nnx.metrics.Accuracy(),
      loss=nnx.metrics.Average('loss'),
    )

    for epoch in range(10):
      for batch in tqdm(train_loader):
        train_step(model, optimizer, metrics, batch)
      print("Epoch:{}_Train Acc@1: {} loss: {} ".format(epoch+1,metrics.compute()['accuracy'],metrics.compute()['loss']))
      metrics.reset()  # Reset the metrics for the train set.

      # Compute the metrics on the test set after each training epoch.
      for test_batch in tqdm(test_loader):
        eval_step(model, metrics, test_batch)
      print("Epoch:{}_Test Acc@1: {} loss: {} ".format(epoch+1,metrics.compute()['accuracy'],metrics.compute()['loss']))
      metrics.reset()  # Reset the metrics for the test set.
