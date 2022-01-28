import numpy as np
from persistence import H1_boundary_matrix, H2_boundary_matrix, rips

X = np.random.uniform(size=(20, 2))
R = rips(X, d=2)


D1 = H1_boundary_matrix(vertices=R['vertices'], edges=R['edges'])
D2 = H2_boundary_matrix(edges=R['edges'], triangles=R['triangles'])

D1 = D1.astype(bool)
D2 = D2.astype(bool)


