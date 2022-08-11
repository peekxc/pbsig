import numpy as np
from pbsig.vineyards import linear_homotopy, line_intersection

n = 500
f0 = np.sort(np.random.uniform(size=n))
f1 = np.random.uniform(size=n)


L = linear_homotopy(f0, f1,  plot_lines=True)

## Validate homotopy
p = np.argsort(f0)
for i,j in L: 
  p[i], p[j] = p[j], p[i]
np.all(np.argsort(f1) == p)

# %% Test circle 
from dms import circle_family
from betti import *
from persistence import * 
C, params = circle_family(12, sd=0.05)
X = C(0.50)

K = rips(X, diam=np.inf, p=2)
ew, tw = rips_weights(X, K['edges']), rips_weights(X, K['triangles'])
K['edges'] = K['edges'][np.argsort(ew)]
K['triangles'] = K['triangles'][np.argsort(tw)]
ew, tw = ew[np.argsort(ew)], tw[np.argsort(tw)]

D1 = boundary_matrix(K, p=1)
D1.data = np.sign(D1.data)*np.repeat(ew, 2)
D2 = boundary_matrix(K, p=2)
D2.data = np.sign(D2.data)*np.repeat(tw, 3)

R1, R2, V1, V2 = reduction_pHcol(D1, D2)

assert validate_decomp(D1, R1, V1, D2, R2, V2)
v,e,t = X.shape[0], len(ew), len(tw)


from persistence import persistence_pairs, validate_decomp
dgm = persistence_pairs(R1, R2, collapse=False)

from persim import plot_diagrams
plot_diagrams(dgm)

from pbsig.vineyards import transpose_dgm, permute_tr
for i in range(e-1):
  D1_copy, R1_copy, V1_copy, D2_copy, R2_copy, V2_copy = D1.copy(), R1.copy(), V1.copy(), D2.copy(), R2.copy(), V2.copy()
  # D1, R1, V1, D2, R2, V2 = D1_copy.copy(), R1_copy.copy(), V1_copy.copy(), D2.copy(), R2_copy.copy(), V2_copy.copy()
  status = transpose_dgm(R1, V1, R2, V2, i)
  permute_tr(D1, i, "cols")
  permute_tr(D2, i, "rows")
  is_valid = validate_decomp(D1, R1, V1, D2, R2, V2)
  print(f"Status: {status}, valid: {is_valid}")
  assert is_valid
  


X = C(1.50)
K = rips(C(0.01), diam=np.inf, p=2)
ew, tw = rips_weights(X, K['edges']), rips_weights(X, K['triangles'])
K['edges'] = K['edges'][np.argsort(ew)]
K['triangles'] = K['triangles'][np.argsort(tw)]
ew, tw = ew[np.argsort(ew)], tw[np.argsort(tw)]

X = C(1.50)
K2 = rips(X, diam=np.inf, p=2)
ew2, tw2 = rips_weights(X, K['edges']), rips_weights(X, K['triangles'])
K2['edges'] = K2['edges'][np.argsort(ew2)]
K2['triangles'] = K2['triangles'][np.argsort(tw2)]
ew2, tw2 = ew2[np.argsort(ew2)], tw2[np.argsort(tw2)]

pairs = persistence_pairs(R1, R2, collapse=False)

birth_edges = ew[pairs[:,0].astype(int)]
tw[pairs[:,1].astype(int)]




# %%
from ripser import ripser
from betti import persistent_betti_rips
T = np.linspace(0.01, 1.5, 1000)
diagrams = [ripser(C(t))['dgms'][1] for t in T]
birth, death = diagrams[int(len(diagrams)/2)][0]*np.array([1.20, 0.80])

# for t in np.linspace(0.01, 1.5, 100):

# Evaluate all four terms exactly 

## Alpha terms for T1 + T2 
ew0 = pdist(C(np.min(T)))
ew1 = pdist(C(np.max(T)))
alpha_e = (birth-ew0)/(ew1-ew0)

## Alpha terms for T3
nv = C(0).shape[0]
TRI = np.array(list(combinations(range(nv), 3)))
tw0 = rips_weights(C(np.min(T)), TRI)
tw1 = rips_weights(C(np.max(T)), TRI)
alpha_t = (death-tw0)/(tw1-tw0)



ALPHA = np.sort(np.append(alpha_e, alpha_t)) + 10*np.finfo(float).resolution

T1 = np.array([np.sum((ew0*(1-a) + ew1*a) <= birth) for a in ALPHA])

D = pdist(C(0.01))
D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)

## T2 
T2 = np.zeros(len(ALPHA), dtype=int)
for i, a in enumerate(ALPHA): 
  ew_alpha = ew0*(1-a) + ew1*a
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew_alpha, 2), 0.0)
  T2[i] = np.linalg.matrix_rank(D1.A)

## T3
T3 = np.zeros(len(ALPHA), dtype=int)
D2_T3 = D2.copy()
for i, a in enumerate(ALPHA): 
  tw_alpha = tw0*(1-a) + tw1*a
  D2_T3.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw_alpha, 3), 0.0)
  T3[i] = np.linalg.matrix_rank(D2_T3.A)

## T4 
T4 = np.zeros(len(ALPHA), dtype=int)
D2_T4 = D2.tocoo().copy()
for i, a in enumerate(ALPHA):
  tw_alpha = (tw0*(1-a) + tw1*a)
  ew_alpha = (ew0*(1-a) + ew1*a)
  D2_T4.data = np.sign(D2.data) * np.minimum(
    np.maximum(death-tw_alpha[D2_T4.col],0.0), 
    np.maximum(ew_alpha[D2_T4.row]-birth,0.0)
  )
  T4[i] = np.linalg.matrix_rank(D2_T4.A)


import matplotlib.pyplot as plt
plt.plot(ALPHA, T1 - T2 - T3 + T4)


plt.plot(ALPHA, -(T2 - T3 + T4))

## Build PL approximation of convex envelope
fvalues = -(T2 - T3 + T4)

from scipy.optimize import optimize

anchors = [0]
a0 = int(anchors[0])
while (a0 < (len(ALPHA)-1)):
  # def f(ai):
  #   ai = int(ai)
  #   #lin_approx = lambda alpha: (1-alpha)*fvalues[a0] + alpha*fvalues[ai]
  #   #valid = np.all(fvalues[a0:(ai+1)] >= (1-ALPHA[a0:(ai+1)])*fvalues[a0] + ALPHA[a0:(ai+1)]*fvalues[ai])
  #   #valid = np.all([lin_approx(a) <= fvalues[a] for a in ])
  #   if valid:
  #     return(0.0)
  #   else:
  #     return(np.abs(ai-a0))
  # a_values = np.array(list(range(a0, len(ALPHA))))
  # nx = np.array([f(a) for a in a_values])
  # np.flatnonzero(np.array([np.all(fvalues[ai] <= fvalues[a0:(ai+1)]) for ai in range(a0+1, len(ALPHA))]))
  anchors.append(a_values[np.max(np.flatnonzero(nx == 0))])
  a0 = anchors[-1]

from scipy.spatial import ConvexHull
ch = ConvexHull(np.c_[ALPHA, fvalues])
plt.plot(ALPHA, -(T2 - T3 + T4))
plt.plot(*np.c_[ALPHA[np.sort(ch.vertices)], fvalues[np.sort(ch.vertices)]].T)

T2_ = -(T2 - T3 + T4)
ch1 = ConvexHull(np.c_[ALPHA, T1])
ch2 = ConvexHull(np.c_[ALPHA, T2_])
plt.plot(*np.c_[ALPHA[np.sort(ch1.vertices)], T1[np.sort(ch1.vertices)]].T)
plt.plot(*np.c_[ALPHA[np.sort(ch2.vertices)], T2_[np.sort(ch2.vertices)]].T)


plt.plot(ALPHA, T1)
plt.plot(ALPHA, -T2)
plt.plot(ALPHA, -T3)
plt.plot(ALPHA, T4)
plt.plot(ALPHA, T1 - T2 - T3 + T4, c='black')
plt.plot(*np.c_[ALPHA[np.sort(ch2.vertices)], T2_[np.sort(ch2.vertices)]].T)


def f1(alpha: float):
  if alpha <= ALPHA[0]:
    return(T1[0])
  elif alpha >= ALPHA[-1]:
    return(T1[-1])
  else:
    return(T1[np.max(np.flatnonzero(ALPHA <= alpha))])

def f2(alpha: float): # absolute valued alpha between ALPHA[0] and ALPHA[-1]
  av = ALPHA[np.sort(ch.vertices)]
  fv = fvalues[np.sort(ch.vertices)]
  if alpha <= av[0]:
    return fv[0]
  elif alpha >= av[-1]:
    return(fv[-1])
  else:
    lb_idx = np.max(np.flatnonzero(av <= alpha))
    rel_alpha = (alpha-av[lb_idx])/(av[lb_idx+1]-av[lb_idx])
    return((1-rel_alpha)*fv[lb_idx] + rel_alpha*fv[lb_idx+1])
  

A = np.linspace(ALPHA[0], ALPHA[-1], 100)

np.array([f2(a) for a in A])

plt.plot(A, np.array([f1(a) for a in A]) - np.array([f2(a) for a in A]))


# max a such that all values ([a0, ai], [f(a0), f(ai)]) lie above (1-lambda)*f(a0) + lambda*f(ai)
# ii, cc = 0, 0
# while ii < len(fvalues):


# %%
