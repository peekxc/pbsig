import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pbsig import * 
from pbsig.datasets import letter_image, freundenthal_image
from pbsig.betti import lower_star_betti_sig
from pbsig.utility import progressbar

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

def persistence_weight(threshold: float = 0.0, p_factor: float = 10.0):
  def _weight(pts: ArrayLike):
    w = 0.0
    for (birth, death) in pts:
      pers = abs(death-birth)
      if pers == np.inf:
        w += 1.0
      else:
        w += 1.0 + pers*p_factor if pers >= threshold else 0.10
    return(w)
  return(_weight)

pw = persistence_weight(threshold=0.10, p_factor=5.0)

from array import array 
dgm0_all = np.concatenate([dgm0 for (dgm0, dgm1) in dgms])
r_ind, c_ind, W = array('I'), array('I'), array('d')
for i, (a,b) in enumerate(dgm0_all):
  in_box = np.logical_and(dgm0_all[:,0] <= a, dgm0_all[:,1] >= b)
  ab_weight = np.sum(pw(dgm0_all[in_box,:])) 
  r_ind.extend(np.flatnonzero(in_box))
  c_ind.extend(np.repeat(i, np.sum(in_box)))
  W.append(ab_weight)

## Approximate the geometric set cover via LP 
cover = coo_matrix((np.ones(shape=(len(r_ind))), (r_ind, c_ind)), shape=(dgm0_all.shape[0], dgm0_all.shape[0])).tocsc()

## Assert 
assert len(np.flatnonzero(cover.sum(axis=1) == 0.0)) == 0
assert len(np.flatnonzero(cover.sum(axis=0) == 0.0)) == 0

## Solve the set cover problem
from pbsig.set_cover import wset_cover_LP, wset_cover_greedy
assignment, cost = wset_cover_LP(cover, W)
# assignment, cost = wset_cover_greedy(cover, W)

## Plot the resulting boxes 
fig = plt.figure(figsize=(3,3), dpi=200)
ax = fig.gca()
ax.set_aspect('equal')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.plot([-1.0, 1.0], [-1.0, 1.0], color='gray',linewidth=0.5, linestyle='-')
ax.scatter(*dgm0_all.T, c=W, cmap='viridis', s=0.15)

best_ind = np.flatnonzero(assignment)[np.argsort(-np.asarray(W)[assignment])]
for (a,b) in dgm0_all[best_ind[:1],:]:
  ax.plot([-1.0, a], [b, b], color='black', linewidth=0.45)
  ax.plot([a, a], [b, 1.0], color='black', linewidth=0.45)

k = 5
best_k = [best_ind[0]]
while len(best_k) < k:
  B = np.setdiff1d(best_ind, best_k)
  int_sz = [len(np.intersect1d(cover[:,b].indices, np.flatnonzero(cover[:,best_k].sum(axis=1)))) for b in B]
  best_k.append(B[np.argmin(int_sz)])

for (a,b) in dgm0_all[best_k,:]:
  ax.plot([-1.0, a], [b, b], color='black', linewidth=0.45)
  ax.plot([a, a], [b, 1.0], color='black', linewidth=0.45)

## --- PARAMETER SELECTION --- 
## Use the parameters to choose the chain relaxation
a, b = dgm0_all[best_ind[0],:] # birth, death box 
sig_params = dict(a=a, b=b, w=max(pdist(X))*0.05, epsilon=0.001)
# sig_params = dict(a=a, b=b, w=1e-12, epsilon=1e-12) # rank version

## Generate the signatures
F = lambda n: progressbar(rotate_S1(X, n, include_direction=False), count=n)
pb0 = lower_star_betti_sig(F(100), p_simplices=E, nv=X.shape[0], **sig_params)
pb1 = lower_star_betti_sig(F(100), p_simplices=T, nv=X.shape[0], **sig_params)

plt.plot(pb0)
plt.plot(pb1)
# plt.gca().set_ylim(-5, 5)

pb_1 = [lower_star_pb(X, T, f, a, b, "rank", terms = False) for f in F(100)]
plt.plot(pb_1)

pb0_curve = np.array([np.sum(np.logical_and(dgm0[:,0] <= a, dgm0[:,1] > b)) for (dgm0, dgm1) in dgms])
pb1_curve = np.array([np.sum(np.logical_and(dgm1[:,0] <= a, dgm1[:,1] > b)) for (dgm0, dgm1) in dgms])

plt.plot(pb1)
plt.plot(pb1_curve)
plt.gca().set_ylim(np.min(pb1_curve)-1, np.max(pb1_curve)+1)

plt.plot(pb0)
plt.plot(pb0_curve)
plt.gca().set_ylim(np.min(pb0_curve)-1, np.max(pb0_curve)+1)

signatures = { l: [] for l in ['A', 'B', 'C', 'D', 'E'] }
for font in ['Lato-Bold', 'OpenSans', 'Ostrich', 'Oswald', 'Roboto']:
  for letter in ['A', 'B', 'C', 'D', 'E']:
    X, E, T = freundenthal_image(letter_image(letter, font=font))
    pb0 = lower_star_betti_sig(F(100), p_simplices=E, nv=X.shape[0], **sig_params)
    pb1 = lower_star_betti_sig(F(100), p_simplices=T, nv=X.shape[0], **sig_params)
    signatures[letter].append(np.c_[pb0, pb1])

plt.plot(signatures[0][:,1], c='blue')
plt.plot(signatures[1][:,1], c='red')
plt.plot(signatures[2][:,1], c='green')
plt.plot(signatures[3][:,1], c='orange')
plt.plot(signatures[4][:,1], c='purple')

from itertools import combinations, product
sig_diffs0 = np.zeros(int(comb(len(signatures.keys())*5,2)))
sig_diffs1 = np.zeros(int(comb(len(signatures.keys())*5,2)))
for ii, ((fs, i), (ft, j)) in enumerate(combinations(product(signatures.keys(), range(5)), 2)):
  sig_diff0 = phase_align(signatures[fs][i][:,0], signatures[ft][j][:,0]) - signatures[ft][j][:,0]
  sig_diff1 = phase_align(signatures[fs][i][:,1], signatures[ft][j][:,1]) - signatures[ft][j][:,1]
  sig_diffs0[ii], sig_diffs1[ii] = np.linalg.norm(sig_diff0), np.linalg.norm(sig_diff1)



from scipy.spatial.distance import squareform 
D = squareform(sig_diffs0)
plt.imshow(D)

plt.imshow(squareform(sig_diffs1))

fonts = ['Lato-Bold', 'OpenSans', 'Ostrich', 'Oswald', 'Roboto']
letter_im = letter_image('A', fonts[2])
plt.imshow(letter_im, cmap="gray")
plot_mesh2D(*freundenthal_image(letter_im))



# from gudhi import bottleneck_distance
# bottleneck_distance(dgms[0][0], dgms[50][0])


# p = 0
# A_00 = signatures['A'][0][:,0]
# np.linalg.norm(phase_align(A_00, signatures['A'][1][:,0]) - signatures['A'][1][:,0])
# np.linalg.norm(phase_align(A_00, signatures['A'][2][:,0]) - signatures['A'][2][:,0])
# np.linalg.norm(phase_align(A_00, signatures['A'][3][:,0]) - signatures['A'][3][:,0])
# np.linalg.norm(phase_align(A_00, signatures['A'][4][:,0]) - signatures['A'][4][:,0])

# np.linalg.norm(phase_align(A_00, signatures['B'][0][:,0]) - signatures['B'][0][:,0])
# np.linalg.norm(phase_align(A_00, signatures['B'][1][:,0]) - signatures['B'][1][:,0])
# np.linalg.norm(phase_align(A_00, signatures['B'][2][:,0]) - signatures['B'][2][:,0])
# np.linalg.norm(phase_align(A_00, signatures['B'][3][:,0]) - signatures['B'][3][:,0])
# np.linalg.norm(phase_align(A_00, signatures['B'][4][:,0]) - signatures['B'][4][:,0])

# np.linalg.norm(phase_align(signatures['A'][0][:,0], signatures['A'][1][:,0]) - signatures['A'][2][:,0])

# # from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator

# # ## Start off with some initial eigen- values / vectors 
# # v1 = np.array([0.0, 1.0])[:,np.newaxis]
# # fv1 = (X @ v1).flatten()
# # D1.data = D1_nz_pattern * np.repeat(fv1[E].max(axis=1), 2)
# # e_val, e_vec = eigsh(D1 @ D1.T, return_eigenvectors=True, which='LM', k=np.min(D1.shape)-1)

# # ## Calculate the cumulative spectra for all angles
# # for fv, v in rotate_S1(X, 100):
# #   D1.data = D1_nz_pattern * np.repeat(fv[E].max(axis=1), 2)
# #   e_val, e_vec = eigsh(D1 @ D1.T, return_eigenvectors=True, which='LM', k=np.min(D1.shape)-1)
# #   total_spec = np.cumsum(-np.sort(-e_val))
# #   plt.plot(total_spec/np.max(total_spec))

# # v2 = np.array([np.cos(0.05*np.pi), np.sin(0.05*np.pi)])[:,np.newaxis]
# # fv2 = (X @ v2).flatten()
# # D1.data = D1_nz_pattern * np.repeat(fv2[E].max(axis=1), 2)

# # ## Truth 
# # eigsh(D1 @ D1.T, return_eigenvectors=False)

# # ## Updated version using shift-invert selections
# # eigsh(D1 @ D1.T, return_eigenvectors=False, v0=e_vec[:,-1], sigma=e_val[-1], which='LM', k=1, maxiter=2)
# # e_val[0]

# # from persim import plot_diagrams
# # for dgm_pair in dgms:
# #   plot_diagrams(dgm_pair)