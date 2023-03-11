import numpy as np 


# %% 
from pbsig.linalg import *
np.random.seed(1234)
S = simplicial_complex([[0,1]])
w0 = np.random.uniform(size=card(S, 0), low=0, high=5.0)
w1 = np.random.uniform(size=card(S, 1), low=0, high=5.0)
w0[0] = 1e-6
D = boundary_matrix(S, p=1)
L = up_laplacian(S, form="array", p=0, weight=np.append(w0,w1), normed=True)
N,n = len(S), card(S,0)

w0[0] = 1e-6
w0_inv = 1.0/np.sqrt(w0)
print(D.todense() @ np.diag(w1) @ D.todense().T)
print(np.diag(w0_inv) @ D.todense() @ np.diag(w1) @ D.todense().T @ np.diag(w0_inv))
np.array(list(reversed(w0_inv))) * w1 * w0_inv

w1 * w0_inv**2

## Ensure the matvec product matches the formula 
# for i,j in combinations(range(n), 2):



## Derivatives of (h * g * f)(a)
L_family = lambda a: np.diag(1/np.sqrt(a*w0)) @ D @ np.diag(a*w1) @ D.T @ np.diag(1/np.sqrt(a*w0))
L_family(0.01).reshape((n*2,1))





L_deriv = Derivative(L_family)
L_deriv(0.001)




from numdifftools import Derivative
w_deriv = Derivative(lambda a: family(a)*w)
deg = np.diag((D @ D.T).todense())
np.diag(w_deriv(0.001)) @ D @ D.T


