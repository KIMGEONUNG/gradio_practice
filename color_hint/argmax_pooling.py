import numpy as np
import skimage
from pycomar.io import load
from pycomar.images import show_img
import cv2.cv2 as cv


def dominant_pool_2d(spatial: np.ndarray, winsize=16):
  """
  Return a 2-D array with a pooling operation.
  The pooling operation is to select the most dominant value for each window.
  This assumes that the input 'spatial' has discrete values like index or lablel.
  To circumvent an use of iterative loop, we use a trick with one-hot encoding
  and 'skimage.measure.block_reduce' function.

  Parameters
  ----------
  spatial : int ndarray of shape (width, hight)
    The spatial is represented by int label, not one-hot encoding
  winsize : int, optional
    Length of sweeping window

  Returns
  -------
  pool : ndarray of shape (N,M)
    The pooling results.

  """
  num_seg = spatial.max() + 1
  one_hot = np.eye(num_seg)[spatial]
  sum_pooling = skimage.measure.block_reduce(one_hot, (winsize, winsize, 1),
                                             func=np.sum)
  pool = np.argmax(sum_pooling, axis=-1)
  return pool


if __name__ == "__main__":
  mask = load("mask.pkl")
  mask = cv.resize(mask, dsize=(16, 16), interpolation=cv.INTER_NEAREST)
  # mask = cv.resize(mask, dsize=(256, 256), interpolation=cv.INTER_NEAREST)
  show_img(mask)

  # print(mask)
  # print(mask.shape)
