import numpy as np
from persistence import *

X = np.random.uniform(size=(20, 2))
R = rips(X, d=2)

D1 = H1_boundary_matrix(vertices=R['vertices'], edges=R['edges'])
D2 = H2_boundary_matrix(edges=R['edges'], triangles=R['triangles'])

# D1 = D1.astype(bool)
# D2 = D2.astype(bool)


R1, R2, V1, V2 = _reduction_naive(D1, D2)

is_reduced(D1)
is_reduced(R2)

piv = np.array([low_entry(R2, j) for j in range(R2.shape[1])], dtype=int)