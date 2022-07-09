import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist, pdist

def dist_to_gram(D):
  '''
  < dist_to_gram >: 
   D := squared distance matrix
  '''
  n = D.shape[0]
  H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
  return(-0.5 * H @ D @ H)


def gram_to_dist(K):
  from itertools import combinations
  n = K.shape[0]
  D = np.zeros(shape=K.shape)
  for i,j in combinations(range(n), 2):
    D[i,j] = D[j,i] = K[i,i] + K[j,j] - 2*K[i,j]
  return(D)

np.allclose(gram_to_dist(dist_to_gram(D)) - D, 0.0)

def cmds(D: ArrayLike, d: int = 2, coords: bool = True, pos: bool = True):
  ''' 
  Computes classical MDS (cmds) 
    D := squared distance matrix
  '''
  n = D.shape[0]
  H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
  evals, evecs = np.linalg.eigh(-0.5 * H @ D @ H)
  evals, evecs = evals[(n-d):n], evecs[:,(n-d):n]

  # Compute the coordinates using positive-eigenvalued components only     
  if coords:               
    w = np.flip(np.maximum(evals, np.repeat(0.0, d)))
    Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
    return(Y)
  else: 
    ni = np.setdiff1d(np.arange(d), np.flatnonzero(evals > 0))
    if pos:
      evecs[:,ni], evals[ni] = 1.0, 0.0
    return(np.flip(evals), np.fliplr(evecs))


theta = np.linspace(0, 2*np.pi, 150, endpoint=False)
X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(len(theta), 2), low=0.01, high=0.05)
X = np.vstack((X, np.random.uniform(size=(50, 2), low=-1.0, high=1.0)))
D = cdist(X, X)**2

# plt.scatter(*X.T)
# ecc = np.linalg.norm(X, axis=1)
from scipy.stats import gaussian_kde
N = gaussian_kde(X.T)
f = N.evaluate(X.T)
f = f / np.linalg.norm(f)

#f = ecc / np.linalg.norm(ecc)
f = f[:,np.newaxis]
P_f = f @ f.T
P_cf = np.eye(len(f)) - P_f
K = dist_to_gram(D)
# np.allclose((Y @ Y.T), K)

# from tallem.samplers import dimred

plt.scatter(*X.T, c=f)

# w,A = cmds(P_cf @ K @ P_cf, d=3, coords=False)
Z = cmds(P_cf @ K @ P_cf, d=3, coords=True)
plt.scatter(*Z.T, c=f)

# ev, U = cmds(D, d=X.shape[0], coords=False)
# np.allclose(U.T @ U, np.eye(X.shape[0]))

A = dist_to_gram(cdist(Z, Z)**2)
np.allclose(A @ f, 0.0)

G = cmds(cdist(np.c_[X, f], np.c_[X, f])**2, d=3, coords=True)
Z = cmds(P_cf @ K @ P_cf, d=3, coords=True)
pdist(G) - pdist(Z)

# cmds(P_cf @ K @ P_cf, d=X.shape[0], coords=False)
import quadprog




## USe Givens rotations


U = np.zeros(shape=(n, n))
U[:,0] = f.flatten()
Q, R = np.linalg.qr(U)

K = double_center(cdist(X, X)**2)



from scipy.optimize import minimize, LinearConstraint
obj = lambda A: np.linalg.norm(K - (f @ f.T + A), 'fro')**2

# minimize(fun, x0, method='SLSQP', constraint=dict(fun=lambda A: 
#   type='eq', 

# )) # COBYLA, SLSQP



X = np.random.uniform(size=(4,2))
D = cdist(X, X)**2
K = double_center(D)


f = np.array([1, 2, 3, 4])[:,np.newaxis]
f = f / np.linalg.norm(f)
P_f = f @ f.T
P_cf = np.eye(len(f)) - P_f
K_f = (K @ P_cf + P_cf @ K)/2



def fcmds(D: ArrayLike, f: ArrayLike, **kwargs):
  f = f[:,np.newaxis] if f.ndim == 1 else f
  P_f = f @ f.T
  P_cf = np.eye(P_f.shape[0]) - P_f
  K = dist_to_gram(D)
  DK = gram_to_dist(P_cf @ K @ P_cf)
  return(cmds(DK**2, **kwargs))


from sklearn import datasets
import matplotlib.pyplot as plt
digits = datasets.load_digits()

# plt.figure(1, figsize=(3, 3))
# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()

Z = cmds(cdist(digits.data,digits.data)**2, 2)

plt.scatter(*Z.T, c=digits.target)

from scipy.cluster.hierarchy import linkage, DisjointSet, leaders, fcluster

#X = np.random.uniform(size=(15, 2))
X = digits.data
L = linkage(pdist(X), method='single')

def merge_dist(L: ArrayLike):
  from itertools import combinations
  W = L[:,2]
  n = L.shape[0]+1
  C = np.array([fcluster(L, w, criterion="distance") for w in W])
  U = np.zeros(shape=(n, n))
  for i,j in combinations(range(n), 2):
    U[i,j] = U[j,i] = W[np.min(np.flatnonzero(C[:,i] == C[:,j]))]
  return(U)

#plt.scatter(*cmds(U, 2).T)
U = merge_dist(L)
# KU = dist_to_gram(U)
uval, uvec = cmds(U**2, d = U.shape[0], coords=False)

ZU = cmds(U**2, d = 2, coords=True)
plt.scatter(*(uvec[:,:2] @ np.diag(np.sqrt(uval[:2]))).T)
plt.scatter(*ZU.T, c=digits.target)

ultrametric_distortion = np.zeros(n)
for d in range(n):
  ZU = uvec[:,:d] @ np.diag(np.sqrt(uval[:d]))
  ultrametric_distortion[d] = np.sum(np.abs(linkage(pdist(ZU), method='single')[:,2] - L[:,2]))

D = cdist(X, X)**2
fcmds_dis = np.zeros(20)
for i in range(20):
  # fuval, fuvec = fcmds(D**2, f=uvec[:,:i], d=D.shape[0], coords=False, pos=False)
  ZU = fcmds(D**2, f=uvec[:,:i], d=D.shape[0], coords=True)
  print(i)
  fcmds_dis[i] = np.sum(np.abs(linkage(pdist(ZU), method='single')[:,2] - L[:,2]))
  #fcmds_dis[i] = np.sqrt(np.sum(np.abs(fuval[fuval < 0]))/n)


plt.plot(fcmds_dis)


Z = cmds(D**2, d=2, coords=True)
plt.scatter(*Z.T, c=digits.target)

Z_u = fcmds(D**2, f=uvec[:,0], d=2, coords=True)

plt.scatter(*Z_u.T, c=digits.target)

# %%  Dendrogram visualization
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import datasets
X = datasets.load_iris().data
I = datasets.load_iris().target

n = X.shape[0]
L = linkage(pdist(X), 'single')
d = dendrogram(L, color_threshold=1, labels=list(range(n)))

from matplotlib import cm
from matplotlib.colors import to_hex
tab20_colors = np.array([to_hex(c) for c in cm.tab20.colors])
class_ids = np.unique(I)
class_color_map = { u : v for u,v in zip(class_ids, tab20_colors[class_ids]) }
class_color_map = {0: 'red', 1: 'green', 2: 'blue'}

d = dendrogram(L, color_threshold=0.0, labels=list(range(n)), link_color_func=lambda k: "black" if k < n else "blue")

#d['leaves_color_list'] = np.array([class_color_map[I[i]] for i in range(n)])

ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
  singleton_id = int(lbl.get_text())
  lbl.set_color(class_color_map[I[singleton_id]])

plt.show()





R = np.random.uniform(low=0.0, high=1.0, size=(150*150))
R[R <= 0.20] = 1
R[R < 1] = 0
R = np.reshape(R, (150,150))
G = ((R + R.T)/2).astype(bool).astype(int)

from scipy.sparse.csgraph import floyd_warshall
D_ambient = floyd_warshall(G)
evals_ambient, evecs_ambient = cmds(D_ambient**2, d=D_ambient.shape[0], coords=False, pos=False)

## Distortion to ambient metric space 
# dis_ambient = 2*np.sqrt(np.sum(np.abs(evals_ambient[evals_ambient < 0]))/n)

ind_pos = np.flatnonzero(evals_ambient > 0)
evecs_ambient[:,ind_pos] @ np.diag(evals_ambient[ind_pos]) @ evecs_ambient[:,ind_pos].T

## Just verify we can reconstruct the ambient 
D_reconstruct = gram_to_dist(evecs_ambient @ np.diag(evals_ambient) @ evecs_ambient.T)
assert np.max(np.abs(D_ambient**2 - D_reconstruct)) < 1e-12

for i in range(len(ind_pos)):
  p = ind_pos[:i]
  D_reconstruct = evecs_ambient[:,p] @ np.diag(evals_ambient[p]) @ evecs_ambient[:,p].T
  print(np.max(np.abs(D_ambient**2 - D_reconstruct)))


## Distortion curve to ambient metric space
nr = np.sum(evals_ambient > 0)
KX = np.zeros(shape=np.shape(D_ambient))
metric_curve = np.zeros(nr)
for i in range(nr):
  KX += evals_ambient[i]*(evecs_ambient[:,[i]] @ evecs_ambient[:,[i]].T)
  RX = gram_to_dist(KX)
  metric_curve[i] = np.linalg.norm(D_ambient**2 - RX, 'fro')

plt.plot(metric_curve)


## Obtain single-linkage matrix on the pairwise graph metric 
dg_pw = Dg[np.triu_indices(n, k=1)]
L_ambient = linkage(dg_pw, method='single')

## square ultrametric distance matrix on ambient graph metric
U_ambient = merge_dist(L)

## Eiegenvalues and eigenvectors for ultrametric
evals_ultra, evecs_ultra = cmds(U_ambient**2, d=n-1, coords=False, pos=False)

## Distortion to ultra-metric space 
dis_ultra = 2*np.sqrt(np.sum(np.abs(evals_ultra[evals_ultra < 0]))/n)

## Print distortion to original ultra-metric as function of k
KU = np.zeros(shape=np.shape(U_ambient))
ultra_curve = np.zeros(n-1)
for i in range(n-1):
  KU += evals_ultra[i]*(evecs_ultra[:,[i]] @ evecs_ultra[:,[i]].T)
  RU = gram_to_dist(KU)
  ultra_curve[i] = np.linalg.norm(U_ambient**2 - RU, 'fro')

## Plot the distortion to the ultra-metric as a function of k
plt.plot(ultra_curve)






fcmds(D, f)


# n = L.shape[0] + 1
# ds = DisjointSet(range(n))
# for cc in range(n-1):
#   i,j = L[cc,[0,1]].astype(int)
#   ds.merge(i, j)
#   if ds.connected(a,b):
#     return(L[cc,2])
# return(L[-1,2])



## 10-d Gaussian blobs 
#centers = np.random.uniform(size=(8, 10), low=-1.0, high=1.0)
from tallem.samplers import landmarks
T = np.random.uniform(low=-1.0, high=1.0, size=(500, 10))
centers = T[landmarks(T, 8)[0],:]

X = np.vstack([np.random.normal(size=(35, 10), loc=c, scale=0.19) for c in centers])
X = np.vstack([X, np.random.uniform(low=-2, high=2, size=(100, 10))])

L = linkage(pdist(X), 'single')
I = np.array([np.repeat(i+1, 35) for i in range(8)]).flatten()
I = np.append(I, np.repeat(0, 100))

Z = cmds(cdist(X,X)**2, d=2)
plt.scatter(*Z.T, c=I)

## Calculate ultra-metric distance matrix
U = merge_dist(L)
uvals, uvecs = cmds(U**2, d=U.shape[0]-1, coords=False, pos=True)
xvals, xvecs = cmds(cdist(X,X)**2, d=X.shape[0], coords=False, pos=True)

ZU = fcmds(cdist(X,X)**2, f=uvecs)

EI = np.floor(np.linspace(0, 5*8, 5*8)).astype(int)
fig, axes = plt.subplots(5, 8, figsize=(18,18), dpi=350)
for vi, (i,j) in zip(EI, product(range(5), range(8))):
  ZU = fcmds(-cdist(X,X)**2, f=uvecs[:,:vi])
  #ZU = fcmds(cdist(U,U)**2, f=xvecs[:,:vi])
  #ZU = fcmds(np.max(cdist(X,X)**2)-cdist(X,X)**2, f=xvecs[:,:vi]) 
  #ZU = fcmds(np.max(cdist(U,U)**2)-cdist(U,U)**2, f=xvecs[:,:vi])
  #ZU = fcmds(np.max(cdist(U,U)**2)/cdist(U,U)**2, f=xvecs[:,:vi])
  axes[i,j].scatter(*ZU.T, c=I, s=0.85)

Z = fcmds(cdist(X,X)**2, f=uvecs[:,:0])
Z2 = cmds(cdist(X, X)**2)
np.allclose(pdist(Z2), pdist(Z))

def plot_dendrogram(L, I):
  from matplotlib import cm
  from matplotlib.colors import to_hex
  n = L.shape[0]+1
  d = dendrogram(L, color_threshold=0.0, labels=list(range(n)), link_color_func=lambda k: "black" if k < n else "blue")
  tab20_colors = np.array([to_hex(c) for c in cm.tab10.colors])
  class_ids = np.unique(I)
  class_color_map = { u : v for u,v in zip(class_ids, tab20_colors[class_ids]) }
  ax = plt.gca()
  xlbls = ax.get_xmajorticklabels()
  for lbl in xlbls:
    singleton_id = int(lbl.get_text())
    lbl.set_color(class_color_map[I[singleton_id]])


plot_dendrogram(L, I)


D, U = cdist(X,X)**2, merge_dist(L)
uvals, uvecs = cmds(U, d=U.shape[0]-1, coords=False)

Z_u = cmds(U, d=2, coords=True)
plt.scatter(*Z_u.T, c=I)

Z_sl = fcmds(D, f=uvecs[:,0])
plt.scatter(*Z_sl.T, c=I)


DZ = cdist(Z_u, Z_u)

UZ = merge_dist(linkage(pdist(Z), 'single'))
UZU = merge_dist(linkage(pdist(Z_u), 'single'))

print(np.linalg.norm(dist_to_gram(U) - dist_to_gram(UZ), 'fro'))
print(np.linalg.norm(dist_to_gram(U) - dist_to_gram(UZU), 'fro'))



cmds
## changing the origin 
n = P.shape[0]



w = np.random.uniform(size=n)[:,np.newaxis]
w = w/np.sum(np.abs(w))

WI = np.array([np.flatnonzero(I == i)[np.argmin(np.linalg.norm(P - P[I == 1, :].mean(axis=0), axis=1))] for i in range(1, 9)])

w = np.ones(n)*0.02/(n-len(WI))
w[WI] = (0.98 / len(WI))
w = w[:,np.newaxis]

s = w.T @ P
IND = np.ones(shape=(n,1))
Pw = (np.eye(n) - IND @ w.T)
Kw = -0.5*(Pw @ cdist(P, P)**2 @ Pw.T)
wvals, wvecs = np.linalg.eigh(Kw)
wvals, wvecs = np.flip(wvals), np.fliplr(wvecs)

WZ = wvecs[:,:2] @ np.diag(np.sqrt(wvals[:2]))
Z = cmds(cdist(P,P)**2, d=2, coords=True)

pdist(Z) - pdist(WZ)

plt.scatter(*WZ.T, c=I)
plt.scatter(*Z.T, c=I)


#Ps = (np.eye(n) - IND @ w.T) @ P

## Constrained MDS 
Y = np.random.uniform(size=(n,3)) # "external" constraints
P, S, Qt = np.linalg.svd(Y, full_matrices=True)
Kx = dist_to_gram(cdist(X, X)**2)
pvals, pvecs = np.linalg.eigh(Kx)
pvals, pvecs = np.flip(pvals), np.fliplr(pvecs)


Zn = (pvecs[:,:2] @  np.diag(1/S[:2]) @ (pvecs[:,:2] @ np.sqrt(np.diag(pvals[:2]))).T) @ Y[:,:2]



np.all(U <= cdist(X, X))