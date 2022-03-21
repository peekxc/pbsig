# %% 
from vineyards import permute_tr, transpose_dgm
from persistence import * 
C, params = circle_family(12, sd=0.05)
X = C(0.50)
# X = np.random.uniform(size=(8,2))

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

v,e,t = X.shape[0], len(ew), len(tw)

dgm = persistence_pairs(R1, R2, f=(ew,tw), collapse=True)

from persim import plot_diagrams
plot_diagrams(dgm)

validate_decomp(D1, R1, V1, D2, R2, V2)

for i in range(e-1):
  D1_copy, R1_copy, V1_copy, D2_copy, R2_copy, V2_copy = D1.copy(), R1.copy(), V1.copy(), D2.copy(), R2.copy(), V2.copy()
  # D1, R1, V1, D2, R2, V2 = D1_copy.copy(), R1_copy.copy(), V1_copy.copy(), D2.copy(), R2_copy.copy(), V2_copy.copy()
  status = transpose_dgm(R1, V1, R2, V2, i)
  permute_tr(D1, i, "cols")
  permute_tr(D2, i, "rows")
  is_valid = validate_decomp(D1, R1, V1, D2, R2, V2)
  print(f"Status: {status}, valid: {is_valid}")
  assert is_valid
  


# %% 
from vineyards import permute_tr, transpose_dgm
from persistence import * 
C, params = circle_family(8, sd=0.10)

# Order doesn't change so just get diagrams and swicth up the function values! 
X = C(0.20)
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
v,e,t = X.shape[0], len(ew), len(tw)

dgm = persistence_pairs(R1, R2, collapse=False).astype(int)

AP = apparent_pairs(pdist(X), K)

## NEED A match function for AP 
e_ranks = rank_combs(K['edges'], k=2, n=len(K['vertices'])).tolist()
t_ranks = rank_combs(K['triangles'], k=3, n=len(K['vertices'])).tolist()
ap_birth = np.array([e_ranks.index(AP[i,0]) for i in range(AP.shape[0])])
ap_death = np.array([t_ranks.index(AP[i,1]) for i in range(AP.shape[0])])

## Now, skip columns R1[:,ap_birth], R2[:,ap_death]
ind_d1 = np.setdiff1d(np.array(range(D1.shape[1])), ap_birth)
# ind_d2 = np.setdiff1d(np.array(range(D2.shape[1])), ap_death)

V1, V2 = sps.identity(D1.shape[1]).tocsc(), sps.identity(D2.shape[1]).tocsc()
R1, R2 = D1.copy(), D2.copy()
pHcol(R1, V1, ind_d1)
pHcol(R2, V2) # have to reduce all of them! 

### Need way of replace R[:,ap_births] OMG SET THEM TO 0
# R1.A[:,[11,15,17,25]]
# R1[:,ap_birth] = 0
  
## These dont match. R1 isn't reduced. Why? 
dgm = persistence_pairs(R1, R2, collapse=False).astype(int)

dgms = []
for t in np.linspace(0, 1.5, 100): 
  X = C(t)
  ew, tw = rips_weights(X, K['edges']), rips_weights(X, K['triangles'])
  dgm_f = np.c_[ew[dgm[:,0].astype(int)], tw[dgm[:,1].astype(int)]]
  dgms.append(dgm_f)

D = np.hstack(dgms)

for i in range(D.shape[0]):
  plt.plot(*D[i,:].reshape((100, 2)).T)