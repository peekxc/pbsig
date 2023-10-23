import numpy as np 
from splex import * 
from splex.geometry import lower_star_filter
from pbsig.linalg import *

S = simplicial_complex([[0,1,2], [1,2,3], [4,5,6,7], [8,9,10], [7,8],[0,7]])
# vf = np.random.uniform(size=card(S,0), low=50, high=150)
# vf = np.random.uniform(size=card(S,0), low=0, high=1e-5)
vf = np.random.uniform(size=card(S,0), low=0, high=1.0)

L = up_laplacian(S, p=1, normed=False)
print(L.todense())
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense())))

L = up_laplacian(S, weight=lower_star_filter(vf), p=1, normed=False)
print(L.todense())
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense())))

## Normalized laplacian should have max spectral norm p+2
L = up_laplacian(S, weight=lower_star_filter(vf), p=1, normed=True)
print(L.todense())
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense())))

## Degree computation 
from scipy.sparse import diags
f = lower_star_filter(vf)
L, d = up_laplacian(S, weight=f, p=1, normed=True, return_diag=True)
wt = np.array([sum(f(t)*f(e) for e in boundary(t)) for t in faces(S, 2)])

e_ranks = rank_combs(faces(S, 1))
deg_e = np.zeros(card(S,1))
for j, t in enumerate(faces(S,2)):
  for e in t.boundary():
    deg_e[e_ranks.index(rank_colex(e))] += wt[j]

from pbsig.linalg import pseudoinverse
D = boundary_matrix(S, p=2)
L = D @ diags(wt) @ D.T 
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense()))) 
L = diags(deg_e) @ D @ diags(wt) @ D.T @ diags(deg_e)
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense()))) 
L = diags(pseudoinverse(np.sqrt(deg_e))) @ D @ diags(wt) @ D.T @ diags(pseudoinverse(np.sqrt(deg_e)))
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense()))) 
## Indeed the degree computation works.


## Does it match the trick
vf[1] = vf[2] = 0.0
f = lower_star_filter(vf)
wt = np.array([sum(f(t)*f(e) for e in boundary(t)) for t in faces(S, 2)])
we = np.array([f(e) for e in faces(S, 1)])
deg_e = np.zeros(card(S,1))
for j, t in enumerate(faces(S,2)):
  for e in t.boundary():
    i = e_ranks.index(rank_colex(e))
    deg_e[e_ranks.index(rank_colex(e))] += wt[j] # the definition 

## edge weights (faces )
from pbsig.linalg import pseudoinverse
print(deg_e)
# deg_test = (diags(we) @ D @ diags(wt) @ D.T @ diags(we)).diagonal()
deg_test = (diags(np.sign(we)) @ D @ diags(wt) @ D.T @ diags(np.sign(we))).diagonal()

np.sort(deg_e)-np.sort(deg_test)

I_norm_true = pseudoinverse(np.sqrt(deg_e))
I_norm_test = pseudoinverse(np.sqrt(deg_test))
L = diags(I_norm_true) @ D @ diags(wt) @ D.T @ diags(I_norm_true)
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense()))) 

L = diags(I_norm_test) @ D @ diags(wt) @ D.T @ diags(I_norm_test)
print("Eigenvalues: ", np.sort(np.linalg.eigvalsh(L.todense()))) 

## They are slightly different, but have the same nullspace

## Indeed the degree computation works. Can it be made continuous?
# from scipy.interpolate import CubicSpline, UnivariateSpline
# from pbsig.interpolate import interpolate_family
# Vf = np.array([np.random.uniform(size=card(S,0), low=0, high=1.0) for i in range(5)])
# Vf[1,1] = Vf[2,2] = Vf[3,1] = 0.0

# vertex_functions = [CubicSpline(np.linspace(0,1,5), Vf[:,i]) for i in range(Vf.shape[1])]



# vf[1:2] = 0.0




  
