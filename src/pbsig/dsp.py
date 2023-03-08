from typing import *
import numpy as np 

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

def phase_align(s1: Sequence, s2: Sequence, return_offset: bool = False):
  """
  Aligns s1 with s2 = rolls s1 to minimize sum of squared pairwise differences between s1 and s2. 

  Returns the newly aligned/permuted s1
  """
  ind = _match_index(s2, s1)
  # fft gets us close; check O(1) offsets in neighborhood for exact rolling 
  offsets = np.fromiter(range(-5, 5), dtype=int)
  r_ind = np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in offsets])
  return(ind+offsets[r_ind] if return_offset else np.roll(s1, ind+offsets[r_ind]))


## Good defaults seems to be scaling=True, center=True, MSE, reverse=True
def signal_dist(a: Sequence[float], b: Sequence[float], method="euc", check_reverse: bool = True, scale: bool = False, center: bool = False) -> float:
  
  ## Center if requested 
  a = a - np.mean(a) if center else a
  b = b - np.mean(b) if center else b

  ## Scale if requested
  normalize = lambda x: 2*((x - min(x))/(max(x) - min(x))) - 1
  a = normalize(a) if (scale and not(np.all(a == a[0]))) else a
  b = normalize(b) if (scale and not(np.all(b == b[0]))) else b

  ## Align the signals by maximizing cross correlation
  d1 = b-phase_align(a,b)
  d2 = b-phase_align(np.flip(a),b) if check_reverse else d1

  ## Compute whatever distance distance 
  if method == "mae":
    d = min(np.mean(np.abs(d1)), np.mean(np.abs(d2)))
  elif method == "mse":
    d = min(np.mean(np.power(d1,2)), np.mean(np.power(d2,2)))
  elif method == "rmse":
    d = np.sqrt(min(np.mean(np.power(d1,2)), np.mean(np.power(d2,2))))
  elif method == "euc":
    d = min(np.sum(np.abs(d1)), np.sum(np.abs(d1)))
  elif method == "convolve":
    base_area = np.trapz(np.convolve(a,a))
    d = np.linalg.norm(base_area - np.trapz(np.convolve(b, phase_align(a,b))))
    d = min(d, np.linalg.norm(base_area - np.trapz(np.convolve(b, phase_align(np.flip(a),b)))))
  else: 
    raise ValueError(f"Invalid distance measure '{method}'")
  return d
  
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