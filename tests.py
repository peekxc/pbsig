import numpy as np
from persistence import *

X = np.random.uniform(size=(20, 2))
R = rips(X, d=2)

D1 = H1_boundary_matrix(vertices=R['vertices'], edges=R['edges'])
D2 = H2_boundary_matrix(edges=R['edges'], triangles=R['triangles'])

# D1 = D1.astype(bool)
# D2 = D2.astype(bool)


R1, R2, V1, V2 = reduction_pHcol(D1, D2)

is_reduced(R1)
is_reduced(R2)
np.sum((D1 @ V1) - R1) == 0.0
np.sum((D2 @ V2) - R2) == 0.0

np.allclose(V1.A, np.triu(V1.A))
np.allclose(V2.A, np.triu(V2.A))

piv = np.array([low_entry(R2, j) for j in range(R2.shape[1])], dtype=int)


R, V = reduction_pHcol_clearing([D2, D1])



## weighted boundary matrices
D1 = H1_boundary_matrix(vertices=R['vertices'], edges=R['edges'])
b = np.max(pdist(X))/10.0
edge_weights = pdist(X)[np.array([rank_C2(u,v,X.shape[0]) for u,v in R['edges']], dtype=int)]

# D1.data = D1.data*(b - np.repeat(edge_weights,2))
D1 = D1[:, np.argsort(edge_weights)]

D1 = D1.astype(float)
eg_val, eg_vec = eigs(D1.T @ D1)
eg_val, eg_vec = eigs(D1 @ D1.T)

# spectral norm contained in [0, d+1]
# for edges (-1, 1) \sigma_max(D1.T @ D1) = \sigma_max(D1 @ D1.T) \in [0, np.min(D1.shape)]

# bounds for spectral norm: 
# 2*np.max(D1) + np.max(np.diag((D1 @ D1.T).A))
[(D1[j,:] @ D1[j,:].T).data.item() for j in range(D1.shape[1])]

# For CSR 
np.max([np.sum(D1[j,:].data**2) for j in range(D1.shape[0])])