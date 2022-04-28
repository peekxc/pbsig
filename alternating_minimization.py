import numpy as np 
from dms import circle_family
from betti import *
from persistence import * 
from scipy.spatial.distance import pdist 
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


C, params = circle_family(n=16, sd=0.15)
T = np.linspace(0, 1.5, 1000)
diagrams = [ripser(C(t))['dgms'][1] for t in T]
birth, death = diagrams[int(len(diagrams)/2)][0]*np.array([1.10, 0.90])

def S(x, b, d):
  if x <= b: return(1)
  if x >= d: return(0)
  return(1 - (x-b)/(d-b))
max_diam = np.max(np.array([np.max(pdist(C(t))) for t in T]))

St = lambda t: np.sum(np.array([S(e, birth, max_diam) for e in ew]))

## Chould be concave + matrix + matrix
def g(D: ArrayLike):
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0) # [:, :i]
  D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0) # [:, :j]
  return(np.array([S(e, birth, max_diam) for e in ew], dtype=float), D1, D2)

nuclear_norm = lambda X: np.sum(np.linalg.svd(X, compute_uv=False))
spectral_norm = lambda X: np.max(np.linalg.svd(X, compute_uv=False))

m0 = np.max([np.max(birth - pdist(C(t))) for t in T])
m1 = np.max([spectral_norm(g(pdist(C(t)))[1].A) for t in T])
m2 = np.max([spectral_norm(g(pdist(C(t)))[2].A) for t in T])

## Normalize using constants 
def g_norm(t: float):
  S0, M1, M2 = g(pdist(C(t)))
  return((1/m1)*nuclear_norm(M1.A) + (1/m2)*nuclear_norm(M2.A) - np.sum((1/m0)*S0))

## Yes, this is convex
plt.plot(T, np.array([g_norm(t) for t in T]))




# T1 = np.zeros(len(T))
# for i, t in enumerate(T):
#   ew = pdist(C(t)) 
#   T1[i] = np.sum(np.array([S(e, birth, max_diam) for e in ew]))
# plt.plot(T, T1)



def h(D: ArrayLike):
  D3, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D3_A, D3_H = D3.tocoo(), D3.tocoo()
  D3_A.data = np.sign(D3.data)*np.maximum(death - tw[D3_A.col], 0.0) 
  D3_H.data = np.sign(D3.data)*np.maximum(ew[D3_H.row] - birth, 0.0)
  return(D3_A.multiply(D3_H).tocsc())

m3 = np.max([spectral_norm(h(pdist(C(t))).A) for t in T])
def h_norm(t: float):
  M3 = h(pdist(C(t)))
  return(-(1/m3)*nuclear_norm(M3.A))


plt.plot(T, [h_norm(t) for t in T])