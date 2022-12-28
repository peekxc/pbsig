from typing import *
import numpy as np 
from scipy import signal

## From: https://stackoverflow.com/a/4696026/6912436
def _rfft_xcorr(x, y):
  M = len(x) + len(y) - 1
  N = 2 ** int(np.ceil(np.log2(M)))
  X = np.fft.rfft(x, N)
  Y = np.fft.rfft(y, N)
  cxy = np.fft.irfft(X * np.conj(Y))
  cxy = np.hstack((cxy[:len(x)], cxy[N-len(y)+1:]))
  return cxy

## From: https://stackoverflow.com/a/4696026/6912436
def _match_index(x, ref):
  cxy = _rfft_xcorr(x, ref)
  index = np.argmax(cxy)
  return index if index < len(x) else index - len(cxy)

def phase_align(s1: Sequence, s2: Sequence):
  """
  Aligns s1 with s2 = rolls s1 to minimize sum of squared pairwise differences between s1 and s2. 
  """
  ind = _match_index(s2, s1)
  # fft gets us close; check O(1) offsets in neighborhood for exact rolling 
  offsets = np.fromiter(range(-5, 5), dtype=int)
  r_ind = np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in offsets])
  return(np.roll(s1, ind+offsets[r_ind]))


# ## From: https://stackoverflow.com/a/4696026/6912436
# def rfft_xcorr(x, y):
#   M = len(x) + len(y) - 1
#   N = 2 ** int(np.ceil(np.log2(M)))
#   X = np.fft.rfft(x, N)
#   Y = np.fft.rfft(y, N)
#   cxy = np.fft.irfft(X * np.conj(Y))
#   cxy = np.hstack((cxy[:len(x)], cxy[N-len(y)+1:]))
#   return cxy

# ## From: https://stackoverflow.com/a/4696026/6912436
# def match(x, ref):
#   cxy = rfft_xcorr(x, ref)
#   index = np.argmax(cxy)
#   return index if index < len(x) else index - len(cxy)

# def phase_align(s1: Sequence, s2: Sequence):
#   ind = match(s2, s1)
#   offsets = np.fromiter(range(-5, 5), dtype=int)
#   r_ind = np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in offsets])
#   return(np.roll(s1, ind+offsets[r_ind]))