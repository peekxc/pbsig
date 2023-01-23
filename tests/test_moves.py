import numpy as np 
import scipy.sparse 
import _persistence as pm
from scipy.sparse import * 

A = scipy.sparse.random(n=10, m=10, density=0.05).tocsc()
F = pm.FloatMatrix(A)

low_entries = [max(A[:,[i]].indices) if A[:,[i]].nnz > 0 else -1 for i in range(A.shape[1])]
assert [F.low(i) for i in range(F.dim[1])] == low_entries

F.clear_column(0)
assert F.low(0) == -1

F.iadd_scaled_col(0,1,1.0)
assert F.low(0) == F.low(1) 

F.swap_cols(2,3)
assert all(F.as_spmatrix()[:,[2]].indices == A[:,[3]].indices)
assert all(F.as_spmatrix()[:,[3]].indices == A[:,[2]].indices)

A = csc_array(np.array([[1,0,1],[-1,1,0],[0,-1,-1]]))
F = pm.FloatMatrix(A)

# while((low_j = R.low(j)) && (piv_i = pivs.at(low_j->first))){ // 
#   const size_t i = piv_i->first; // column i of R must have low row index low_j->first 
#   const field_t lambda = low_j->second / piv_i->second;
#   R.iadd_scaled_col(j, i, -lambda); 	// zeros pivot in column j
#   V.iadd_scaled_col(j, i, -lambda);   // col(j) <- col(j) + s*col(i)
F.iadd_scaled_col(2,1, -(-1.0/-1.0))


F.permute_rows([0,1,2])



from pbsig.datasets import random_lower_star
from pbsig.simplicial import SimplicialComplex
X, K = random_lower_star(10)

from pbsig.persistence import *
D = boundary_matrix(K)
V = eye(D.shape[0])
I = np.arange(0, D.shape[1])
R, V = pm.phcol(D, V, I)

assert is_reduced(R)
assert np.isclose((R - (D @ V)).sum(), 0.0)

S = SimplicialComplex([[0],[1],[2],[0,1],[0,2],[1,2]])
K = MutableFiltration(S)
D = boundary_matrix(K)
V = eye(D.shape[0])
I = np.arange(0, D.shape[1])
R, V = pm.phcol(D, V, I)
assert is_reduced(R) and np.isclose((R - (D @ V)).sum(), 0.0)

## Test move right in python 
from pbsig.vineyards import move_right

print(R.todense())
print(V.todense())
Rm, Vm = pm.move_right(R, V, 3, 5)
print(Rm.todense())
print(Vm.todense())

