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