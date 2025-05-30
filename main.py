 # optax for optimizer
import optax
# The Flax NNX API.
from flax import nnx 
# grain to load data
import grain.python as grain
# orbax to save model
import orbax.checkpoint as ocp
from pathlib import Path
ckpt_dir = Path(Path.cwd() / './checkpoints')
# model 
from net import CNN
# load data
from feeder import load_data

def loss_fn(model: CNN, batch):
  _, logits = model(batch['data'])
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
  _, logits = model(batch['data'])
  return logits.argmax(axis=1)

if __name__ == '__main__':
  # hyperparameters
  batch_size = 256
  
  # Load the data.
  train_loader, test_loader = load_data('snr', num_workers=8, num_epochs=10, batch_size=batch_size)
  
  # Instantiate the model.
  model = CNN(rngs=nnx.Rngs(0), num_classes=3)
  optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-6,b1=0.9))
  metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
  )

  best_acc = 0
  train_steps = int(train_loader._sampler._num_epochs * train_loader._sampler._num_records / 256)
  checkpointer = ocp.StandardCheckpointer()
  for step, batch in enumerate(train_loader):
    train_step(model, optimizer, metrics, batch)
    if step > 0 and (step % 500 == 0 or step == train_steps - 1):
      print("Step:{}_Train Acc@1: {} loss: {} ".format(step,metrics.compute()['accuracy'],metrics.compute()['loss']))
      metrics.reset()  # Reset the metrics for the train set.

      # Compute the metrics on the test set after 500 training steps.
      for test_batch in test_loader:
        eval_step(model, metrics, test_batch)
      print("Step:{}_Test Acc@1: {} loss: {} ".format(step,metrics.compute()['accuracy'],metrics.compute()['loss']))
      if metrics.compute()['accuracy'] > best_acc:
        best_acc = metrics.compute()['accuracy']
        _, state = nnx.split(model)
        checkpointer.save(ckpt_dir / 'best_snr', state)
      metrics.reset()  # Reset the metrics for the test set.