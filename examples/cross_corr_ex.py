from typing import Sequence
import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt


tau = np.linspace(0, 2*np.pi, 100, endpoint=False)

## Example 1: aligning sin + noisy sin
n = len(tau)
s1 = np.sin(tau)
s2 = np.sin(tau + np.pi*(5/4)) + np.random.uniform(low=-0.025, high=0.025, size=len(tau))

c1 = signal.correlate(s1, s1, mode='same')[int(n/2)]
c2 = signal.correlate(s2, s2, mode='same')[int(n/2)]

C = signal.correlate(s1, s2, mode='same')
best_ind = 

np.linspace(-0.5, 0.5, n)[np.argmax(C/np.sqrt(c1*c2))]

C = signal.correlate(s1, s2, mode='full')
max_ind = np.argmax(np.abs(C))
delay = tau[max_ind - (len(tau)-1)]

plt.plot(tau, s1)
plt.plot(tau - delay, s2)

from scipy.fft import fft, ifft
af = fft(s1)
bf = fft(s2)
c = ifft(af * np.conj(bf))

time_shift = np.argmax(abs(c))

plt.plot(tau, s1, c='orange')
plt.plot(tau, s2, c='blue')
# plt.plot(tau + np.pi*(5/4), s2)
plt.plot(tau, np.concatenate([s2[time_shift:], s2[:time_shift]]), c='red')
# plt.plot(tau+2*np.pi, np.concatenate([s2[:time_shift], s2[time_shift:]]), c='red')

## From: https://stackoverflow.com/a/4696026/6912436
def rfft_xcorr(x, y):
  M = len(x) + len(y) - 1
  N = 2 ** int(np.ceil(np.log2(M)))
  X = np.fft.rfft(x, N)
  Y = np.fft.rfft(y, N)
  cxy = np.fft.irfft(X * np.conj(Y))
  cxy = np.hstack((cxy[:len(x)], cxy[N-len(y)+1:]))
  return cxy

## From: https://stackoverflow.com/a/4696026/6912436
def match(x, ref):
  cxy = rfft_xcorr(x, ref)
  index = np.argmax(cxy)
  if index < len(x):
    return index
  else: # negative lag
    return index - len(cxy)   

def phase_align(s1: Sequence, s2: Sequence):
  ind = match(s2, s1)
  offsets = np.fromiter(range(-5, 5), dtype=int)
  r_ind = np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in offsets])
  return(np.roll(s1, ind+offsets[r_ind]))
  #np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in range(-5, 5)])

from scipy.signal import sawtooth

s1 = sawtooth(tau + np.pi/2, width=1)
s2 = sawtooth(tau + np.pi*(5/4)) + np.random.uniform(low=-0.025, high=0.025, size=len(tau))
plt.plot(tau, s1, c='orange')
plt.plot(tau, s2, c='blue')
plt.plot(tau, phase_align(s1, s2), c='red')
# np.argmin([np.linalg.norm(s2 - np.roll(s1, ind+i)) for i in range(-5, 5)])

# L = tau[-1] - tau[0]
# i_max = np.argmax(signal.correlate(s1, s2, mode='same'))
# phi_shift = np.linspace(-0.5*L, 0.5*L , N)
#     N = len(t)
#     L = t[-1] - t[0]
    
#     cc = signal.correlate(x, y, mode="same")
#     i_max = np.argmax(cc)
#     phi_shift = np.linspace(-0.5*L, 0.5*L , N)
#     delta_phi = phi_shift[i_max]

# np.argmax(C)



## Example 2
s1 = np.sin(tau)
s2 = np.cos(tau)
#s3 = s1*np.sqrt(abs(s2))


p1 = np.sin(np.pi/2 + tau) + np.random.uniform(low=-0.025, high=0.025, size=len(tau))
p2 = np.cos(np.pi/2 + tau) + np.random.uniform(low=-0.025, high=0.025, size=len(tau))
#p3 = s1*np.sqrt(abs(s2)) + np.random.uniform(low=-0.025, high=0.025, size=len(tau))


plt.plot(tau, s1)
plt.plot(tau, s2)
# plt.plot(tau, s3)

plt.plot(tau, p1)
plt.plot(tau, p2)

C = signal.correlate(np.c_[s1, s2], np.c_[p1, p2], mode='full').shape

C = signal.correlate(np.c_[s1, s2], np.c_[p1, p2], mode='same')

plt.plot(tau, s1)
plt.plot(tau, s2)
plt.plot(tau, p1)
plt.plot(tau, p2)
plt.plot(tau, C[:,0])
plt.plot(tau, C[:,1])

plt.plot(tau, normalize(s2))
plt.plot(tau, normalize(C[:,0]))

normalize = lambda x: (x - min(x))/(max(x) - min(x))