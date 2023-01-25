import numpy as np 
import scipy.sparse 
import _persistence as pm
from scipy.sparse import * 
from pbsig.persistence import *

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



def test_move_right():
  for cj in range(100):
    np.random.seed(cj)
    X, K = random_lower_star(25)
    D = boundary_matrix(K)
    V = eye(D.shape[0])
    I = np.arange(0, D.shape[1])
    R, V = pm.phcol(D, V, I)
    assert is_reduced(R)
    assert np.isclose((R - (D @ V)).sum(), 0.0)
    D, R, V = D.astype(int).tolil(), R.astype(int).tolil(), V.astype(int).tolil()

    ## Try random movements that respect the face poset
    S = list(K.values())
    valid_mr = []
    for i,j in combinations(range(len(S)), 2):
      respects_face_poset = all([not(S[i] <= s) for s in S[(i+1):(j+1)]])
      if not(respects_face_poset): # comment out to try non-face moves
        valid_mr.append((i,j))
    
    cc = np.argmax(np.diff(np.array(valid_mr), axis=1))
    i, j = valid_mr[cc]
    # print(S[i])
    # print(S[(i+1):(j+1)])
    #plt.spy(R)
    assert is_reduced(move_right(R, V, i, j, copy=True)[0])
    #move_right(R, V, i, j, copy=True)  
    move_right(R, V, i, j, copy=False)  
    #plt.spy(R)

    idx_set = np.arange(len(S))
    rind = np.roll(np.arange(i,j+1), -1)
    bind = np.arange(len(S))
    bind[i:(j+1)] = rind
    S = [S[i] for i in bind]
    K = MutableFiltration(zip(idx_set, S))
    PDP = boundary_matrix(K)  
    # plt.spy(PDP @ V)

    assert is_reduced(R)
    assert np.allclose(((PDP @ V).todense() - R.todense()) % 2, 0)
  

from pbsig.vineyards import *
def test_move_left():
  for cj in range(100):
    np.random.seed(cj)
    X, K = random_lower_star(25)
    D = boundary_matrix(K)
    V = eye(D.shape[0])
    I = np.arange(0, D.shape[1])
    R, V = pm.phcol(D, V, I)
    assert is_reduced(R)
    assert np.isclose((R - (D @ V)).sum(), 0.0)
    D, R, V = D.astype(int).tolil(), R.astype(int).tolil(), V.astype(int).tolil()

    ## Try random movements that respect the face poset
    S = list(K.values())
    valid_ml = []
    for i,j in combinations(range(len(S)), 2):
      respects_face_poset = not(any([s <= S[j] for s in S[i:j]]))
      if not(respects_face_poset):
        valid_ml.append((i,j))
    
    cc = np.argmax(np.diff(np.array(valid_ml), axis=1))
    i, j = valid_ml[cc]
    # print(S[j])
    # print(S[i:j])
    # plt.spy(R)
    assert is_reduced(move_left(R, V, j, i, copy=True)[0])
    #move_right(R, V, i, j, copy=True)  
    move_left(R, V, j, i, copy=False)  
    #plt.spy(R)


    P = permutation_matrix(move_left_permutation(i,j, len(K)))
    R = R.todense() % 2
    V = V.todense() % 2 
    PDP = (P @ D @ P.T).todense() % 2
    assert is_reduced(R)
    assert is_reduced((PDP @ V) % 2)
    assert np.allclose(((PDP @ V) - R) % 2, 0)
    print(cj)