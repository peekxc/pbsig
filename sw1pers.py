# %% Module imports
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# %% Function imports
from typing import *
from numpy.typing import ArrayLike
from dms import *
from persistence import * 
from tallem.dimred import pca
from persistence import sliding_window
from betti import relaxed_pb, nuclear_constants

# %% Choose the function to optimize the slidding window
f = lambda t: np.cos(t) + np.cos(3*t)
# f = lambda t: np.cos(t)

# %% Plot periodic function
dom = np.linspace(0, 12*np.pi, 1200)
plt.plot(dom, f(dom))

# %% Make a slidding window embedding
N, M = 120, 24
F = sliding_window(f, bounds=(0, 12*np.pi))
X_delay = F(n=N, d=M, L=6)
plt.scatter(*pca(X_delay).T)
plt.gca().set_aspect('equal')

# %% Use ripser to infer range of [b, d]
from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(F(n=N, d=M, L=6))['dgms']
lifetimes = np.diff(diagrams[1], axis=1).flatten()
h1_bd = diagrams[1][np.argmax(lifetimes),:]
print(f"Most persistent H1 cycle: { h1_bd }")
birth, death = h1_bd[0]*1.20, h1_bd[1]*0.20

plot_diagrams(diagrams, show=False)
plt.scatter(birth, death, s=8.5, c='red', zorder=10)
plt.hlines(y=death, xmin=-100, xmax=birth, colors='black', linestyles='-', lw=1)
plt.vlines(x=birth, ymin=death, ymax=100, colors='black', linestyles='-', lw=1)

# %% Enumerate the Persisent Betti numbers
d,tau_min = sw_parameters(bounds=(0, 12*np.pi), d=M, L=24)
d,tau_max = sw_parameters(bounds=(0, 12*np.pi), d=M, L=1)
T = np.linspace(0.40, 0.60, 100)
PBS = np.array([persistent_betti_rips(F(n=N, d=M, tau=tau), b=birth, d=death, summands=True) for tau in T])
PB = PBS[:,0] - (PBS[:,1] + PBS[:,2])

# %% A. Compute relaxation constants 
g = lambda t: F(n=N, d=M, tau=t)
m1, m2 = nuclear_constants(N, birth, death, "tight", g, T)
# m1, m2 = nuclear_constants(N, birth, death, "global", g, T)

# %% B. Compute relaxation
PBRS = np.array([relaxed_pb(g(t), b=birth, d=death, bp=birth+0.50*(death-birth), m1=m1, m2=m2, summands=True, rank_adjust=False, p=(1.0, 1.0)) for t in T])
PBR = PBRS[:,0] - (PBRS[:,1] + PBRS[:,2])

assert np.all(PBS[:,2] >= PBRS[:,2])
# i = np.argmax(np.abs(PBS[:,2]-PBRS[:,2]))
# i = np.flatnonzero(PBS[:,2] < PBRS[:,2])[0]
# PBRS[i,:]
# PBS[i,:]

# %% Overall plot - looks nothing like objective
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), dpi=350)
ax1.plot(T, PB, c='black', label='f')
ax1.plot(T, PBS[:,0], c='yellow', label='S')
ax1.plot(T, PBS[:,1], c='orange', label='T1')
ax1.plot(T, PBS[:,2], c='red', label='T2')
ax1.legend()
ax2.plot(T, PBR, c='blue', label='Relaxed f', zorder=30)
# ax2.plot(T, PBR*(PB+1), c='green', label='squared')
ax2.scatter(T, PBR, c='black', zorder=31, s=0.5)
ax2.plot(T, PBRS[:,0], c='yellow', label='S')
ax2.plot(T, PBRS[:,1], c='orange', label='T1')
ax2.plot(T, PBRS[:,2], c='red', label='T2')
ax2.legend()

# %% Individual term plots
fig, AX = plt.subplots(2, 3, figsize=(12,5), dpi=350)
AX[0,0].plot(T, PBS[:,0], label='Betti S')
AX[0,0].plot(T, PBRS[:,0], label='Relaxed Betti S')
AX[0,0].legend()
AX[0,1].plot(T, PBS[:,1], label='Betti T1')
AX[0,1].plot(T, PBRS[:,1], label='Relaxed Betti T1')
AX[0,1].legend()
AX[0,2].plot(T, PBS[:,2], label='Betti T2')
AX[0,2].plot(T, PBRS[:,2], label='Relaxed Betti T2')
AX[0,2].legend()

AX[1,0].plot(T, np.abs(PBS[:,0]-PBRS[:,0]), label='S Error (L1)')
AX[1,0].legend()
AX[1,1].plot(T, np.abs(PBS[:,1]-PBRS[:,1]), label='T1 Error (L1)')
AX[1,1].legend()
AX[1,2].plot(T, np.abs(PBS[:,2]-PBRS[:,2]), label='T2 Error (L1)')
AX[1,2].legend()

# %% 
plt.plot(T, PB, c='black', label='Relaxed f', zorder=30)
plt.plot(T, PBR, c='blue')

# %% Show relaxation of identity -> constant
from betti import relax_identity
dom = np.linspace(0, 1, 100)
P = np.exp(np.linspace(np.log(1), np.log(100), 20))
for p in P:
  dom = np.linspace(0, 1, 100)
  rng = relax_identity(dom, p, negative=False)
  plt.plot(dom, rng)

# %% Gradient verification
# from betti import pb_gradient
# GS = np.array([pb_gradient(f, t, b, d, SG, summands=True) for t in T])



# %% Bezier experiments 
import bezier
nodes = np.asfortranarray([
  [0.0, 0.0, 1.0], 
  [0.0, 1.0, 1.0]
], dtype=float)
bc = bezier.Curve(nodes, degree=2)

from scipy.optimize import root_scalar

def relax_identity2(X, p: float = 1.0):
  return(np.array([bc.evaluate(np.sqrt(x**(1/p)))[1] for x in X]))
  # return(np.array([root_scalar(lambda t: bc.evaluate(t**p)[0] - x, bracket=(0.0, 1.0)).root for x in X]))
  # return(np.array([bc.evaluate(root_scalar(lambda t: bc.evaluate(t**p)[0] - x, bracket=(0.0, 1.0)).root)[1] for x in X]))

plt.plot(dom, np.array([bc.evaluate(np.sqrt(x)**p)[1] for x in X]))
plt.plot(dom, relax_identity2(dom, p))

dom = np.linspace(0, 1, 1000)
P = np.exp(np.linspace(np.log(1), np.log(250), 25, endpoint=True))
for d in np.linspace(0, 1, 25, endpoint=True):
  # p = np.exp(d*np.log(250))
  #p = np.log(d + 1)*np.log(250)
  # plt.plot(dom, relax_identity(dom, p))
  plt.plot(dom, relax_identity(dom, d, negative=True))# just use this



