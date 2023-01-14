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
G_vals = np.array([g_norm(t) for t in T])
plt.plot(T, G_vals)




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

nuc_vals = np.array([h_norm(t) for t in T])
plt.plot(T, [h_norm(t) for t in T])
plt.plot(T, [spectral_norm(h(pdist(C(t))).A) for t in T])

from betti import soft_threshold, prox_moreau

# mu1 in [0, 1], mu2 in [0, infinity]
def moreau_M3(t: float, mu1: float, mu2: float):
  alpha = (1/m3)
  mu_alpha = mu1*(1/alpha) # mu should be in [0, 1/alpha], otherwsie matrix is 0 
  M3 = h(pdist(C(t)))
  usv = np.linalg.svd(M3.A, full_matrices=False)
  s = np.maximum(usv[1]-mu_alpha, 0.0)
  prox_M3 = alpha*(usv[0] @ np.diag(s) @ usv[2])
  return(alpha*np.sum(s) + (1/(2*mu2))*np.linalg.norm(M3 - prox_M3, 'fro'))
  #print(len(M3.data))

# moreau = lambda t: prox_moreau(h(pdist(C(t))).A, alpha=(1/m3), mu=10.50*(1/m3))[0]
for p in np.exp(np.linspace(np.log(1e-6 + 1), np.log(0.50), 20))-1:
  M_vals = np.array([moreau_M3(t, p, 0.50) for t in T])
  #M_vals = np.array([moreau(t) for t in T])
  plt.plot(T, M_vals)

plt.plot(T, G_vals)
plt.plot(T, G_vals - M_vals)


M3_val = lambda t: moreau_M3(t, 1e-3, 0.80)

M3_vals = np.array([M3_val(t) for t in T])

plt.plot(T, M3_vals)
JM3 = np.array([jac_finite_diff(t, M3_val) for t in T])
plt.plot(T, JM3)
crit_id = np.flatnonzero([np.sign(JM3)[i] != np.sign(JM3)[i+1] for i in range(len(JM3)-1)])
plt.scatter(T[crit_id], JM3[crit_id], c='red')


plt.plot(T, -M3_vals)
plt.scatter(T[crit_id], M3_vals[crit_id], c='red')

lb, ub = 0.58, 0.82
T = np.linspace(lb, ub, 1000)

G_vals = np.array([g_norm(t) for t in T])
M3_vals = np.array([M3_val(t) for t in T])

plt.plot(T, G_vals)
plt.plot(T, M3_vals)







f = lambda t: np.cos(t) + np.cos(3*t)
dom = np.linspace(0, 12*np.pi, 1200)
plt.plot(dom, f(dom))

# %% Make a slidding window embedding
M = 24
N = 50
F = sliding_window(f, bounds=(0, 12*np.pi))
def C(t: float):
  X_delay = F(n=N, d=M, tau=t)
  return(X_delay)

# %% Use ripser to infer range of [a,d]
from ripser import ripser
from persim import plot_diagrams
diagrams = ripser(F(n=N, d=M, L=6))['dgms']
lifetimes = np.diff(diagrams[1], axis=1).flatten()
h1_bd = diagrams[1][np.argmax(lifetimes),:]
birth, death = h1_bd[0]*1.20, h1_bd[1]*0.80
print(f"Most persistent H1 cycle: { h1_bd }; a={birth}, b={death}")

d,tau_min = sw_parameters(bounds=(0, 12*np.pi), d=M, L=24)
d,tau_max = sw_parameters(bounds=(0, 12*np.pi), d=M, L=1)


# T = np.linspace(tau_min, tau_max, 100)
T = np.linspace(1.0001, 1.05, 100)

from scipy.sparse.linalg import svds
def nuclear_norm_sp(A, deflate: bool = True):
  A_deflated = A[:,np.ediff1d(A.indptr) > 0] if deflate else A
  s = svds(A_deflated,k=np.min(A_deflated.shape)-1, return_singular_vectors=False)
  return(np.sum(s))

def spectral_norm_sp(A, deflate: bool = True):
  A_deflated = A[:,np.ediff1d(A.indptr) > 0] if deflate else A
  s = svds(A_deflated,k=1, return_singular_vectors=False)
  return(s.item())

# spectral_norm = lambda A: np.max(np.linalg.svd(A, compute_uv=False))
# spectral_norm_sp = lambda A: svds(A, k=1, return_singular_vectors=False).item()
m1 = 0.0
for t in T: 
  D = pdist(C(t))
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0)
  D1.eliminate_zeros()
  m1 = np.max([m1, spectral_norm_sp(D1)])

m2 = 0.0
for t in T: 
  D = pdist(C(t))
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False) 
  D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0)
  D2.eliminate_zeros()
  m2 = np.max([m2, spectral_norm_sp(D2)])

m0 = np.max([np.max(birth - pdist(C(t))) for t in T])

def S(x, b, d):
  if x <= b: return(1)
  if x >= d: return(0)
  return(1 - (x-b)/(d-b))
max_diam = np.max(np.array([np.max(pdist(C(t))) for t in T]))

St = lambda t: np.sum(np.array([S(e, birth, max_diam) for e in ew]))

def g(D: ArrayLike):
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0) # [:, :i]
  D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0) # [:, :j]
  D1.eliminate_zeros()
  D2.eliminate_zeros()
  return(np.array([S(e, birth, max_diam) for e in ew], dtype=float), D1, D2)

## Normalize using constants 
def g_norm(t: float):
  S0, M1, M2 = g(pdist(C(t)))
  return((1/m1)*nuclear_norm_sp(M1) + (1/m2)*nuclear_norm_sp(M2) - np.sum((1/m0)*S0))

G_vals = np.array([g_norm(t) for t in T])
plt.plot(T, G_vals)

import timeit
from scipy.sparse.linalg import svds

timeit.timeit(lambda: svds(D2, k=1, return_singular_vectors=False))
svd()


from tallem.dimred import landmarks
Lind, _ = landmarks(C(0.5), 50)
rips_boundary(pdist(C(0.5)[Lind,:]), p=2, diam=r)

