import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pbsig import * 
from pbsig.datasets import letter_image, freundenthal_image
from scipy.sparse import coo_matrix

## Load the letters images

A_img = letter_image('A')

## Plot the letter 
plt.imshow(A_img, cmap="gray")

## Compute the Freundental triangulation to get an embedded simplicial complex
X, E, T = freundenthal_image(A_img)

## Plot triangulation 
plot_mesh2D(X, E, T)

## Generate a pair of diagrams (dgm0, dgm1) from multiple viewing angles along S1
from pbsig.persistence import lower_star_ph_dionysus
dgms = [lower_star_ph_dionysus(f=fv, E=E, T=T) for fv, v in rotate_S1(X, 100)]

## Plot the persistence diagrams on top of each 
fig = plt.figure(figsize=(3,3), dpi=200)
ax = fig.gca()
ax.set_aspect('equal')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.plot([-1.0, 1.0], [-1.0, 1.0], color='gray',linewidth=0.5, linestyle='-')
for (dgm0, dgm1) in dgms:
  ax.scatter(*dgm0.T, c='red', s=0.15)

## Build the sparse set cover representation
from array import array 
dgm0_all = np.concatenate([dgm0 for (dgm0, dgm1) in dgms])
r_ind, c_ind, W = array('I'), array('I'), array('d')
for i, (a,b) in enumerate(dgm0_all):
  in_box = np.logical_and(dgm0_all[:,0] <= a, dgm0_all[:,1] >= b)
  ab_weight = np.sum([1.0 if p == np.inf else 1.0 + p*10 for p in dgm0_all[in_box,1] - dgm0_all[in_box,0]]) 
  r_ind.extend(np.flatnonzero(in_box))
  c_ind.extend(np.repeat(i, np.sum(in_box)))
  W.append(ab_weight)

## Approximate the geometric set cover via LP 
cover = coo_matrix((np.ones(shape=(len(r_ind))), (r_ind, c_ind)), shape=(dgm0_all.shape[0], dgm0_all.shape[0]))

## Assert 
assert len(np.flatnonzero(cover.sum(axis=1) == 0.0)) == 0
assert len(np.flatnonzero(cover.sum(axis=0) == 0.0)) == 0

## Solve the set cover problem
from pbsig.set_cover import wset_cover_LP, wset_cover_greedy
assignment, cost = wset_cover_LP(cover, np.asarray(W))
# assignment, cost = wset_cover_greedy(cover, np.asarray(W))

## Plot the resulting boxes 
fig = plt.figure(figsize=(3,3), dpi=200)
ax = fig.gca()
ax.set_aspect('equal')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.plot([-1.0, 1.0], [-1.0, 1.0], color='gray',linewidth=0.5, linestyle='-')
for (dgm0, dgm1) in dgms: ax.scatter(*dgm0.T, c='red', s=0.15)

best_ind = np.flatnonzero(assignment)[np.argsort(-np.asarray(W)[assignment])]
for (a,b) in dgm0_all[best_ind[:1],:]:
  ax.plot([-1.0, a], [b, b], color='black', linewidth=0.75)
  ax.plot([a, a], [b, 1.0], color='black', linewidth=0.75)


## --- PARAMETER SELECTION --- 
## Use the parameters to choose the chain relaxation
a, b = dgm0_all[best_ind[0],:] # birth, death box 
w = 0.0000001         # size of the window around the birth/death indices
epsilon = 0.00001     # how close the rank approximation should be

## Make the smooth step functions centered at (a,b)
from pbsig.utility import smoothstep
machine_eps = np.finfo(float).eps
ss_a = smoothstep(lb = a - w/2, ub = a + w/2, reverse = True)
ss_ac = smoothstep(lb = a - w/2, ub = a + w/2, reverse = False)
ss_b = smoothstep(lb = b - w/2, ub = b + w/2, reverse = True)
# ss_ae = smoothstep(lb = (a + machine_eps) - w/2, ub = (a + machine_eps) + w/2, reverse = True)
# plt.plot(np.linspace(a - w, a + w, 100), ss_a(np.linspace(a - w, a + w, 100)))

## Now actually compute the betti relaxation
from pbsig.betti import lower_star_boundary
nv = X.shape[0]
D1, ew = lower_star_boundary(np.repeat(1.0, nv), simplices=E)
D2, tw = lower_star_boundary(np.repeat(1.0, nv), simplices=T)
D1_nz_pattern, D2_nz_pattern = np.sign(D1.data), np.sign(D2.data)

shape_sig = array('d')
for fv, v in rotate_S1(X, 100):
  
  ## Term 1
  T1 = ss_a(fv[E].max(axis=1))

  ## Term 2 
  D1.data = D1_nz_pattern * ss_a(np.repeat(fv[E].max(axis=1), 2))
  T2 = eigsh(D1.T @ D1, return_eigenvectors=False, k=min(D1.shape)-1)
  
  ## Term 3
  t_chain_val = ss_b(fv[T].max(axis=1))
  D2.data = D2_nz_pattern * np.repeat(t_chain_val, 3)
  D2.data[D2.data == 0.0] = 0.0 # fixes sign issues ? 
  T3 = eigsh(D2.T @ D2, return_eigenvectors=False, k=min(D2.shape)-1)
  T3[T3 < 1e-13] = 0.0

  ## Term 4
  d2_01, d2_02, d2_12 = fv[T[:,[0,1]]].max(axis=1), fv[T[:,[0,2]]].max(axis=1), fv[T[:,[1,2]]].max(axis=1)
  sa = ss_ac(np.ravel(np.vstack((d2_01, d2_02, d2_12)).T))
  sb = np.repeat(t_chain_val, 3)
  D2.data = D2_nz_pattern * (sa * sb)
  D2.data[D2.data == 0.0] = 0.0
  T4 = eigsh(D2.T @ D2, return_eigenvectors=False, k=min(D2.shape)-1)
  T4[abs(T4) < 1e-13] = 0.0

  ## TODO: incorporate tolerance: x > np.max(<s_vals>)*np.max(D*.shape)*np.finfo(D*.dtype).eps
  terms = (np.sum(T1/(T1 + epsilon)), np.sum(T2/(T2 + epsilon)), np.sum(T3/(T3 + epsilon)), np.sum(T4/(T4 + epsilon)))
  shape_sig.append(np.sum(terms))

# lower_star_pb(X, T, fv, a, b, "rank", terms = True)



from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator

## Start off with some initial eigen- values / vectors 
v1 = np.array([0.0, 1.0])[:,np.newaxis]
fv1 = (X @ v1).flatten()
D1.data = D1_nz_pattern * np.repeat(fv1[E].max(axis=1), 2)
e_val, e_vec = eigsh(D1 @ D1.T, return_eigenvectors=True, which='LM', k=np.min(D1.shape)-1)

## Calculate the cumulative spectra for all angles
for fv, v in rotate_S1(X, 100):
  D1.data = D1_nz_pattern * np.repeat(fv[E].max(axis=1), 2)
  e_val, e_vec = eigsh(D1 @ D1.T, return_eigenvectors=True, which='LM', k=np.min(D1.shape)-1)
  total_spec = np.cumsum(-np.sort(-e_val))
  plt.plot(total_spec/np.max(total_spec))

v2 = np.array([np.cos(0.05*np.pi), np.sin(0.05*np.pi)])[:,np.newaxis]
fv2 = (X @ v2).flatten()
D1.data = D1_nz_pattern * np.repeat(fv2[E].max(axis=1), 2)

## Truth 
eigsh(D1 @ D1.T, return_eigenvectors=False)

## Updated version using shift-invert selections
eigsh(D1 @ D1.T, return_eigenvectors=False, v0=e_vec[:,-1], sigma=e_val[-1], which='LM', k=1, maxiter=2)
e_val[0]

# from persim import plot_diagrams
# for dgm_pair in dgms:
#   plot_diagrams(dgm_pair)