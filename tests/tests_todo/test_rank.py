import numpy as np


D1 = np.array([[0.1, -0.1, 0], [0.2, 0.0, -0.2], [0.0, 0.3, -0.3]]).T
E = np.array([[0.025, -0.025, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).T

D = D1.T @ D1

A = D1 + E

np.sum(np.linalg.eigh(A.T @ A)[0])
np.sum(np.linalg.eigh(D + E.T @ E)[0])

W = abs(D - A.T @ A) # difference 
u = np.array([1.0, 0.0, 0.0])[:,np.newaxis]
v = (np.array([0.5, 1.0, 1.0]) * W[:,0])[:,np.newaxis]

np.trace(np.linalg.inv(D + 0.0001*np.eye(3)) @ D)
np.sum(np.linalg.eigh(D)[0]/(np.linalg.eigh(D)[0] + 0.0001))


np.sum(np.linalg.eigh(D + u @ v.T + v @ u.T)[0])

c1 = np.sqrt(np.linalg.norm(v)/(2*np.linalg.norm(u)))
c2 = np.linalg.norm(v)/np.linalg.norm(u)
x, y = c1*(u+c2*v), c1*(u-c2*v)

## this one
c1 = np.linalg.norm(v)/(2*np.linalg.norm(u))
c2 = np.linalg.norm(u)/np.linalg.norm(v)
x, y = u + c2*v, u - c2*v
c1*(x @ x.T) - c1*(y @ y.T)
u @ v.T +  v @ u.T

## Yes this works
np.sum(np.linalg.eigh(D + u @ v.T +  v @ u.T)[0])
np.sum(np.linalg.eigh(A.T @ A)[0])



np.sum(np.linalg.eigh(D)[0])

np.sum(np.diag(D)[:2])-np.sum(np.diag(D)[:1])

np.sum(np.diag(D)[:2])-np.sum(np.diag(D)[:1])
np.sum(np.diag(D)[:1])-np.sum(np.diag(D)[:0])



D1 = np.random.uniform(size=(10,10))
D = D1.T @ D1

W = np.array([np.sum(np.diag(D1[:,:(i+1)].T @ D1[:,:(i+1)])) for i in range(10)])
W[9]-W[8]

[np.sum(np.diag(D1[:,:(i+1)].T @ D1[:,:(i+1)])) for i in range(10)]

np.array([np.sum(np.diag(D)[:(10-i)])-np.sum(np.diag(D)[:(9-i)]) for i in range(10)])

np.sum(np.linalg.eigh(D)[0])








## Goal: Fix a sparse matrix of some size (m x n)
## Then, given K1 < K2 < ... < Kn with updated filtration values f(s1) < f(s2) < ... < f(sn)
## Update (Dp.T @ Dp) very quickly using the same non-zero pattern



