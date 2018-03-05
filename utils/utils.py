import numpy as np
from skimage.morphology import label
from collections import OrderedDict
from tensorflow.python.framework import ops


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def get_tensor_aliases(tensor):
  """Get a list with the aliases of the input tensor.

  If the tensor does not have any alias, it would default to its its op.name or
  its name.

  Args:
    tensor: A `Tensor`.

  Returns:
    A list of strings with the aliases of the tensor.
  """
  if hasattr(tensor, 'aliases'):
    aliases = tensor.aliases
  else:
    if tensor.name[-2:] == ':0':
      # Use op.name for tensor ending in :0
      aliases = [tensor.op.name]
    else:
      aliases = [tensor.name]
  return aliases


def convert_collection_to_dict(collection, clear_collection=False):
  """Returns an OrderedDict of Tensors with their aliases as keys.

  Args:
    collection: A collection.
    clear_collection: When True, it clears the collection after converting to
      OrderedDict.

  Returns:
    An OrderedDict of {alias: tensor}
  """
  output = OrderedDict((alias, tensor)
                       for tensor in ops.get_collection(collection)
                       for alias in get_tensor_aliases(tensor))
  if clear_collection:
    ops.get_default_graph().clear_collection(collection)
  return output