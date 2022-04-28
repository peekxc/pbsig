import numpy as np 
from dms import circle_family
from betti import *
from persistence import * 
from scipy.spatial.distance import pdist 
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# %% circle data set 
C, params = circle_family(n=16, sd=0.15)
T = np.linspace(0, 1.5, 1000)
diagrams = [ripser(C(t))['dgms'][1] for t in T]
birth, death = diagrams[int(len(diagrams)/2)][0]*np.array([1.10, 0.90])

# %% Show persistence diagram
DGM = np.vstack(diagrams)
fig = plt.figure(figsize=(3,3), dpi=150)
ax = plt.gca()
ax.plot(*np.vstack(diagrams).T)
ax.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), c='black')
ax.set_aspect('equal')
ax.scatter(birth, death,c='red',s=1.0)
ax.hlines(y=death, xmin=0, xmax=birth, colors='orange', lw=0.5)
ax.vlines(x=birth, ymin=death, ymax=3.0, colors='orange', lw=0.5)
feasible = np.flatnonzero(np.logical_and(DGM[:,0] <= birth, DGM[:,1] > death))+1 # + 1 because first is trash

from scipy.linalg import block_diag
def f(t, sep=False):
  D = pdist(C(t))
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0)
  
  ## 
  #D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0)
  #D2[ew > birth,:] = 0

  ## Top-left 
  D2 = D2.tocoo()
  b_ei = np.maximum(birth - ew[D2.row], 0.0)
  d_ej = np.maximum(death - tw[D2.col], 0.0)
  D2.data = np.sign(D2.data)*np.minimum(b_ei, d_ej)
  return((D1, D2) if sep else block_diag(D1.A, D2.A))

## Check convex(concave) envelope of T2 
nuclear_norm = lambda X: np.sum(np.linalg.svd(X, compute_uv=False))
spectral_norm = lambda X: np.max(np.linalg.svd(X, compute_uv=False))
constants = np.array([spectral_norm(f(t)) for t in T])
m = np.max(constants)
envelope = np.array([nuclear_norm(f(t)) for t in T])
plt.plot(-(1/m)*envelope) # sure enough, this is concave!

## Check concavity of indicator relaxation
def S(x, b, d):
  if x <= b: return(1)
  if x >= d: return(0)
  return(1 - (x-b)/(d-b))

## Important! 
max_diam = np.max(np.array([np.max(pdist(f(t))) for t in T]))
T1 = np.zeros(len(T))
for i, t in enumerate(T):
  ew = pdist(C(t)) 
  ## Relaxed version
  # T1[i] = np.sum(np.array([S(e, birth, death) for e in ew]))
  
  ## Exact version (not concave!)
  #T1[i] = np.sum(ew <= birth)
  
  ## Relaxed version (should be concave for all values)
  # T1[i] = np.sum(np.array([S(e, birth*2.50, max_diam) for e in ew]))
  T1[i] = np.sum(np.array([S(e, birth, max_diam) for e in ew]))
plt.plot(T, T1)

## Problem: normalization of the sum
plt.plot(T, T1 + -(1/m)*envelope)
plt.plot(T, T1 + -(1/constants)*envelope, color='g')
plt.axvline(x=T[np.min(feasible)], color='r')
plt.axvline(x=T[np.max(feasible)], color='r')

h = (np.max(T)-np.min(T))/250
grad = np.zeros(len(T))

for i,t in enumerate(T): 
  Jt = (f(t + h) - f(t - h))/(2*h)
  usv = np.linalg.svd(f(t), full_matrices = False)
  G = (usv[0] @ usv[2])
  gt = np.reshape(G, (1, np.prod(G.shape)))
  jt = np.reshape(Jt, (np.prod(Jt.shape), 1))
  grad[i] = gt @ jt

import matplotlib.pyplot as plt
plt.plot(T, grad)

grad_g = np.zeros(len(T))
for i, t in enumerate(T):
  ew = pdist(C(t)) 
  grad_g[i] = np.sum(np.logical_and(ew >= birth, ew <= death))


plt.plot(T, grad_g)
plt.plot(T, grad)
plt.plot(T, grad_g + 0.025*grad)

X = C(np.median(T))
D1, D2, D3 = persistent_betti_rips_matrix(X, birth, death)
D1.data = np.sign(D1.data)*np.maximum(birth - abs(D1.data), 0)

## Apparent pairs idea
from apparent_pairs import apparent_pairs
from persistence import *
R = rips(C(T[10]), np.inf, p = 2)
AP = apparent_pairs(pdist(C(np.median(T))), R)

envelope_ap = np.zeros(len(T))
for i, t in enumerate(T):
  lex_ew = pdist(C(t))
  pivot_cols = AP[lex_ew[AP[:,0]] > birth,1]
  A, CD = f(t, sep=True)
  better_cols = np.setdiff1d(np.array(range(CD.shape[1])), pivot_cols)
  envelope_ap[i] = nuclear_norm(block_diag(A.A, CD.A[:,better_cols]))

plt.plot(T, T1 + -(1/m)*envelope)
plt.plot(T, T1 + -(1/constants)*envelope, color='g')
plt.plot(T, T1 + -(1/m)*envelope_ap, color='r')
plt.axvline(x=T[np.min(feasible)], color='r')
plt.axvline(x=T[np.max(feasible)], color='r')

#tri_pivots = unrank_combs(AP[:,1], k=3, n=C(0).shape[0])


## Maxima bounding
T1 = np.array([np.sum(pdist(C(t)) <= birth) for t in T])
ZF = T[np.flatnonzero(T1 == np.max(T1))]

T2 = np.array([np.linalg.matrix_rank(f(t)) for t in T])
ZG = 

np.argmax(T1 + T2)
plt.plot(T1 + T2)


import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdastr, lambdify
from sympy.solvers import solve
import numpy
import matplotlib.pyplot as plt

# http://www.mcduplessis.com/index.php/2016/10/18/automatically-calculating-the-convex-conjugate-fenchel-dual-using-pythonsympy/
def calc_conjugate(str, varnames='x'):
  # set the symbols
  vars = sp.symbols(varnames)
  x = vars[0] if isinstance(vars, tuple) else vars
  y = sp.symbols('y', real=True)

  # set the function and objective
  fun = parse_expr(str)
  obj = x*y - fun

  # calculate derivative of obj and solve for zero
  sol = solve(sp.diff(obj, x), x)

  # substitute solution into objective
  solfun = sp.simplify(obj.subs(x, sol[0]))

  # if extra values were passed add to lambda function
  varnames = [y] + list(vars[1:]) if isinstance(vars, tuple) else y

  return (sp.sstr(solfun), lambdify(vars, fun, 'numpy'), lambdify(varnames, solfun, 'numpy'))

print('function: {0} conjugate {1}'.format('x**2 - 5', calc_conjugate('x**2 - 5')[0]))

calc_conjugate('log(1 + exp(x))')

## This should be the S function
b, d = 30, 40
mu = 1/(d-b)
dom = np.linspace(0, d, 1000)
f = lambda x: 1 - mu*np.log(1 + np.exp(x-b))
plt.plot(dom, f(dom))
plt.gca().set_ylim(bottom=0, top=1.05)

## Make convex
f = lambda x: -(1 - mu*np.log(1 + np.exp(x-b)))
plt.plot(dom, f(dom))
plt.gca().set_ylim(bottom=-1.05, top=1.05)

s, fr, fc = calc_conjugate('-(1-z*log(1 + exp(x-b)))') # y*log(-y*exp(b)/(y - z)) - z*log(-z/(y - z)) + 13e3e
fconj = lambda y: y*np.log(-y*np.exp(b)/(y - mu)) - mu*np.log(-mu/(y - mu)) + 1
eps = np.finfo(float).eps
cdom = np.linspace(eps, mu-eps, 1000)
plt.plot(cdom, fconj(cdom))
#plt.gca().set_aspect('equal')
#plt.gca().set_ylim(bottom=-1.05, top=1.05)

## 

print('function: {0} conjugate {1}'.format(funstr, fconj_str))


## 
x = numpy.linspace(-5, 5, 1000)
y = numpy.linspace(-5, 5, 1000)
funstr = 'log(1+exp(x-10))'
fconj_str, flam, fconjlam = calc_conjugate(funstr)
fx = flam(x)
fconjy = fconjlam(y)
# remove the undefined values
idx = numpy.isfinite(fx)
x = x[idx]
fx = fx[idx]
idx = numpy.isfinite(fconjy)
y = y[idx]
fconjy = fconjy[idx]
fig = plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(x, fx, color='red', linewidth=2.0)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim([min(x), max(x)])
plt.title(funstr)
plt.subplot(122)
plt.plot(y, fconjy, color='black', linewidth=2.0)
plt.xlabel('y')
plt.ylabel(r'f*(y)')
plt.xlim([min(y), max(y)])
fig.subplots_adjust(bottom=0.2)



d = 10
mu1 = 1/(c)
f = lambda x: 1 - 0.01*np.log(1 + np.exp(x-c))
plt.plot(dom, f(dom))
plt.gca().set_ylim(bottom=0, top=1.05)

calc_conjugate('k*log(1 + l*exp(x))')
z = 0.05
f_conj = lambda y: y*(np.log(-y/(y - 1))) - np.log(-1/(y - 1))
# f_conj = lambda y: y*(np.log(-y/(y + z)) + 5) + z*np.log(z/(y + z)) - 1
# f_conj = lambda y: y*(np.log(-y/(y - z)) + 5) - z*np.log(-z/(y - z))
cdom = np.linspace(0.01, 1-0.01, 1000)
plt.plot(cdom, f_conj(cdom))
plt.gca().set_ylim(bottom=0, top=1.05)

# s, f, fc = calc_conjugate('z*log(1 + exp(x-5))')


# %% DC program idea
from numpy.typing import ArrayLike

def gh(D: ArrayLike):
  """
  D := pairwise distances @ t
  """
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D3 = D2.copy()
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0) # [:, :i]
  D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0) # [:, :j]
  D3 = D3.tocoo()
  D3.data = np.sign(D3.data)*np.minimum(np.maximum(death - tw[D3.col], 0.0), np.maximum(ew[D3.row] - birth, 0.0))
  return(D1, D2, D3.tocsc())

### NORMALIZING CONSTANTS DONT FORGET
spectral_norm = lambda A: np.max(np.linalg.svd(A, compute_uv=False))
d1,d2,d3 = gh(pdist(C(np.max(T))))
m1,m2,m3 = spectral_norm(d1.A), spectral_norm(d2.A), spectral_norm(d3.A)

def g(D: ArrayLike):
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D3 = D2.copy()
  D1.data = (1/m1)*np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0) # [:, :i]
  D2.data = (1/m2)*np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0) # [:, :j]
  return(D1, D2)

def h(D: ArrayLike):
  D3, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D3 = D3.tocoo()
  D3.data = (1/m3)*np.sign(D3.data)*np.minimum(np.maximum(death - tw[D3.col], 0.0), np.maximum(ew[D3.row] - birth, 0.0))
  # D3.data = (1/m3)*np.sign(D3.data)*np.maximum(ew[D3.row] - birth, 0.0)
  return(D3.tocsc())


## S function, its conjugate, and its conjugate's gradient
d = np.max(pdist(C(np.max(T)))) # not death!
b = birth
mu = 1/(d-b)
dom = np.linspace(0, d, 1000)
f = lambda x: 1 - mu*np.log(1 + np.exp(x-b))
plt.plot(dom, f(dom))

from autograd import grad
from autograd import numpy as auto_np
def S_conj(y: float) -> float:
  return(y*auto_np.log(-y*auto_np.exp(b)/(y - mu)) - mu*auto_np.log(-mu/(y - mu)) + 1)

## Effective domain: (0, mu)
S_conj_grad = grad(S_conj)
vec = lambda A: np.reshape(A, (np.prod(A.shape), 1))


h_eps = np.sqrt(np.finfo(float).resolution)*100
Xk, Yk, N = [], [], 100
t, cc = 1.4, 0 
iterates = []


while (cc < N):

  ## grad h 
  ## what if h(t) = 0?
  D3, D3_p, D3_m = h(pdist(C(t))), h(pdist(C(t + h_eps))), h(pdist(C(t - h_eps)))
  D3_jac = (D3_p - D3_m)/(2*h_eps)
  usv3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False)
  grad_h = (vec(usv3[0] @ usv3[2]).T @ vec(D3_jac)).item()

  #t_new = t + 0.001*grad_h
  # t_new = np.minimum(np.maximum(0.0+h_eps, 0.00001*grad_h), np.max(T))
  t_new = t + np.sign(grad_h)*0.01
  D1, D2 = g(pdist(C(t_new)))
  D1_p, D2_p = g(pdist(C(t_new + h_eps)))
  D1_m, D2_m = g(pdist(C(t_new - h_eps)))
  D1_jac = (D1_p - D1_m)/(2*h_eps)
  D2_jac = (D2_p - D2_m)/(2*h_eps)
 
  ## Subgradient of  g^*(y)
  usv1 = np.linalg.svd(D1.A, compute_uv=True, full_matrices=False) 
  uv1 = usv1[0][:,[0]] @ usv1[2][[0],:]
  grad_d1 = (vec(uv1).T @ vec(D1_jac.A)).item()

  usv2 = np.linalg.svd(D2.A, compute_uv=True, full_matrices=False) 
  uv2 = usv2[0][:,[0]] @ usv2[2][[0],:]
  grad_d2 = (vec(uv2).T @ vec(D2_jac.A)).item()

  ew = pdist(C(t_new))
  y_diam = (ew/d)*mu - np.finfo(float).eps
  grad_SC = np.sum(np.array([S_conj_grad(y) for y in y_diam]))
  
  grad_g = 0.00001*(grad_d1 + grad_d2 - grad_SC)
  #t = np.minimum(np.maximum(0.0+h_eps, grad_g), np.max(T))
  t = t_new + np.sign(grad_g)*0.01

  iterates.append([t_new, t, cc])
  cc += 1

# T[feasible] \in [0.67, 8.2]

I = np.vstack(iterates)

plt.plot(I[:,2], I[:,0])
plt.plot(I[:,2], I[:,1])

nuclear_norm = lambda X: np.sum(np.linalg.svd(X, compute_uv=False))
hv = np.array([nuclear_norm(h(pdist(C(t))).A) for t in T])
plt.plot(T, hv)

hv = np.array([nuclear_norm(h(pdist(C(t))).A) for t in T])
# hv = np.array([spectral_norm(h(pdist(C(t))).A) for t in T])
#hv = np.array([nuclear_norm(h(pdist(C(t))).A) for t in T])
plt.plot(T, hv)


gv1 = np.array([nuclear_norm(g(pdist(C(t)))[0].A) for t in T])
gv2 = np.array([nuclear_norm(g(pdist(C(t)))[1].A) for t in T])
plt.plot(T, gv1)
plt.plot(T, gv2)


## Testing autograd jacbian vs finite difference
# def D1_obj(t: float):
#   D = pdist(C(t))
#   D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
#   D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0) # [:, :i]
#   return(D1.A)

# %% Log-det approach 

## Make the blog matrix 
def sd_blk(X: ArrayLike):
  usv = np.linalg.svd(X, compute_uv=True, full_matrices=False)
  Y = usv[0] @ usv[0].T
  Z = usv[2].T @ usv[2]
  B = np.hstack((np.vstack((Y, X.T)), np.vstack((X, Z))))
  return(B)
# log det(A) = sum log(\sigma_i(A))
# np.sum(np.log(np.linalg.svd(B + (1.1)*np.eye(B.shape[0]), compute_uv=False)))
# np.linalg.slogdet(B+(1.1)*np.eye(B.shape[0]))

def all_three(A):
  if np.allclose(A, 0):
    ld, nn, mr = 0, 0, 0
  else: 
    ld = np.abs((np.linalg.slogdet(sd_blk(A) + 0.70*np.eye(np.sum(A.shape)))[1]))
    nn = nuclear_norm(A)
    mr = np.linalg.matrix_rank(A)
  return([ld, nn, mr])

# np.trace((G.T @ G).A) != ?
#m = spectral_norm(g(pdist(C(np.max(T))))[0].A)
#m = spectral_norm(h(pdist(C(np.max(T)))).A)
res = np.zeros(shape=(len(T), 10))
for i,t in enumerate(T): 
  X = pdist(C(t))
  D1, D2 = g(X)
  D3 = h(X)
  res[i,:] = np.append([np.sum(X <= birth)], np.array([all_three(D1.A), all_three(D2.A), all_three(D3.A)]).flatten())
  
normalize = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
plt.plot(T, normalize(res[:,2]))
plt.plot(T, normalize(res[:,1]))
plt.plot(T, normalize(res[:,0]))


plt.plot(T, res[:,0])

plt.plot(T, -(res[:,1] + res[:,4]) + res[:,7])

plt.plot(T, res[:,0] -(res[:,1] + res[:,4]) + res[:,7])

plt.plot(T, res[:,0] - res[:,3] - res[:,6] + res[:,9])

plt.plot(T, res[:,2])
plt.plot(T, res[:,2])







# %% DC attempt 2
# def circle_family(n: int, lb: float = 0.0, ub: float = np.inf, sd: float = 0):
#   theta = auto_np.linspace(0, 2*auto_np.pi, n, endpoint=False)
#   unit_circle = auto_np.c_[auto_np.cos(theta), auto_np.sin(theta)]
#   unit_circle += auto_np.random.normal(scale=sd, size=unit_circle.shape)
#   def circle(t: float):
#     t = auto_np.max([t, 0])
#     return(auto_np.dot(unit_circle, auto_np.diag([t, t])))
#     #return(auto_np.sum(auto_np.abs(auto_np.dot(unit_circle, auto_np.diag([t, t])))))
#   return(circle)
# C = circle_family(n=16, sd=0.15)


### NORMALIZING CONSTANTS DONT FORGET
nuclear_norm = lambda A: np.sum(np.linalg.svd(A, compute_uv=False))
spectral_norm = lambda A: np.max(np.linalg.svd(A, compute_uv=False))

def g(D: ArrayLike): # unnormalized
  D1, (vw, ew) = rips_boundary(D, p=1, diam=np.inf, sorted=False) 
  D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D1.data = np.sign(D1.data)*np.maximum(birth - np.repeat(ew, 2), 0.0)
  D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0) 
  return(D1, D2)

def h(D: ArrayLike): # unnormalized
  D3, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
  D0 = csc_matrix(np.diag(np.maximum(birth - ew, 0.0)))
  D3 = D3.tocoo()
  # D3.data = np.sign(D3.data)*np.minimum(np.maximum(death - tw[D3.col], 0.0), np.maximum(ew[D3.row] - birth, 0.0))
  D3.data = np.sign(D3.data)*np.maximum(ew[D3.row] - birth, 0.0)
  return(D0, D3.tocsc())

m0 = np.max([np.max(birth - pdist(C(t))) for t in T])
m1 = np.max([spectral_norm(g(pdist(C(t)))[0].A) for t in T])
m2 = np.max([spectral_norm(g(pdist(C(t)))[1].A) for t in T])
m3 = np.max([spectral_norm(h(pdist(C(t)))[1].A) for t in T])

## Normalize using constants 
g_norm = lambda t: tuple(np.array([1/m1, 1/m2])*g(pdist(C(t))))
h_norm = lambda t: tuple(np.array([1/m0, 1/m3])*h(pdist(C(t))))

## This is indeed convex with the single rule
plt.plot(T, np.array([nuclear_norm(((1/m3)*h(pdist(C(t)))[1]).A) for t in T]))

## This is convex
t2 = np.array([np.sum([nuclear_norm(A.A) for A in h_norm(t)]) for t in T])
plt.plot(T, t2)

## This is convex
t1 = np.array([np.sum([nuclear_norm(A.A) for A in g_norm(t)]) for t in T])
plt.plot(T, t1)

## This is not convex, but should be a DC
plt.plot(T, t1-t2)
plt.axvline(x=T[np.min(feasible)], color='r')
plt.axvline(x=T[np.max(feasible)], color='r')

# Generalized central finite-difference Jacobian for a vector-valued function f: R -> R^m
# f must return something overloaded to with arithmetic operators (+,-,/,*)
def jac_finite_diff(t, f: Callable, n: int = 2, eps = 1e-3):
  """
  Uses an 2n+1 point central finite difference approximation
  """
  from math import comb
  assert n >= 1 or n <= 4, 'n must be in [1, 4]'
  if n == 1: 
    R = (f(t+eps)-f(t-eps))/(2*eps)
  elif n == 2: 
    R = (f(t-2*eps)-8*f(t-eps)+8*f(t+eps)-f(t+2*eps))/(12*eps)
  elif n == 3: 
    R = (-f(t-3*eps)+9*f(t-2*eps)-45*f(t-eps)+45*f(t+eps)-9*f(t+2*eps)+f(t+3*eps))/(60*eps)
  elif n == 4: 
    R = (3*f(t-4*eps)-32*f(t-3*eps)+168*f(t-2*eps)-672*f(t-eps)+672*f(t+eps)-168*f(t+2*eps)+32*f(t+3*eps)-3*f(t+4*eps))/(840*eps)
  else: 
    raise ValueError("Invalid n")
  return(R)

def tanh(x):
  y = auto_np.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

plt.plot(np.linspace(-7, 7, 200), tanh(np.linspace(-7, 7, 200)))

jac_finite_diff(1.0, tanh, n=4, eps=h)

(1-tanh(1.0)**2)

## Autograd'd rips
from itertools import combinations
nv = C(0).shape[0]
LR2 = np.array([rank_combs([[i,j], [i,k], [j,k]], k=2, n=nv) for (i,j,k) in combinations(range(nv),3)])

pdist(C(0.5))[LR2.flatten()]

import autograd.numpy as auto_np
auto_np.max(pdist(C(0.5))[LR2], axis=1)

A = np.reshape(X, (X.shape[0], X.shape[1], 1))
B = np.reshape(X, (X.shape[0], 1, X.shape[1]))


G = X.T @ X 
vec(np.diag(G)).T + vec(np.diag(G)) - 2*G


def auto_pdist(t: float):
  X = C(t).T
  A = auto_np.reshape(X, (X.shape[0], X.shape[1], 1))
  B = auto_np.reshape(X, (X.shape[0], 1, X.shape[1]))
  G = auto_np.dot(X.T, X)
  d = auto_np.reshape(auto_np.diag(G), (G.shape[0], 1))
  D = d.T + d - 2*G
  return(D[auto_np.triu_indices(X.shape[1], k=1)])

def auto_tdist(D: ArrayLike):
  from itertools import combinations
  nv = inverse_choose(len(D), 2)
  nt = int((nv*(nv-1)*(nv-2))/(2*3))
  td = auto_np.zeros(nt)
  for cc, (i,j,k) in enumerate(combinations(range(nv), 3)):
    ij, ik, jk = rank_C2(i, j, nv), rank_C2(i, k, nv), rank_C2(j, k, nv)
    td[cc] = auto_np.max(D[[ij, ik, jk]])
  return(auto_np.sum(td))

def h_grad(t):
  D0, D3 = h_norm(t)
  D0_jac = jac_finite_diff(t, lambda t: h_norm(t)[0], n=2)
  D3_jac = jac_finite_diff(t, lambda t: h_norm(t)[1], n=2)
  usv3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False)
  grad_d0 = np.sum(D0_jac)
  grad_d3 = (vec(usv3[0] @ usv3[2]).T @ vec(D3_jac)).item()
  grad_h = grad_d0 + grad_d3
  return(grad_h)

h_grad_T = np.array([h_grad(t) for t in T])

def g_grad(t):
  D1, D2 = h_norm(C(t))
  D1_jac = jac_finite_diff(t_new, lambda t: g(C(t))[0], n=2)
  D2_jac = jac_finite_diff(t_new, lambda t: g(C(t))[1], n=2)

DC_subgradient(t, h_grad, g_grad)


Xk, Yk, N = [], [], 100
t, cc = 0.5, 0 
iterates = []

while (cc < N):

  ## grad(h(t))
  D0, D3 = h_norm(t)
  D0_jac = jac_finite_diff(t, lambda t: h_norm(t)[0], n=2)
  D3_jac = jac_finite_diff(t, lambda t: h_norm(t)[1], n=2)
  usv3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False)

  grad_d0 = np.sum(D0_jac)
  grad_d3 = (vec(usv3[0] @ usv3[2]).T @ vec(D3_jac)).item()
  grad_h = grad_d0 + grad_d3

  # from autograd import grad
  # fs = lambda t: auto_np.sum((1/m0)*auto_np.maximum(birth - auto_pdist(t), 0.0))
  # dg = grad(lambda t: auto_np.sum((1/m0)*auto_np.maximum(birth - auto_pdist(t), 0.0)))
  # gs = np.array([dg(t) for t in np.linspace(1e-5, 1.5, 100)])
  # gs2 = np.array([jac_finite_diff(t, fs, n=3) for t in np.linspace(1e-5, 1.5, 100)])
  # plt.plot(np.linspace(1e-5, 1.5, 100), gs)
  # plt.plot(np.linspace(1e-5, 1.5, 100), gs2)


  #t_new = t + 0.001*grad_h
  # t_new = np.minimum(np.maximum(0.0+h_eps, 0.00001*grad_h), np.max(T))
  t_new = grad_h
  D1, D2 = g_norm(t_new)
  D1_jac = jac_finite_diff(t_new, lambda t: g_norm(t)[0], n=2)
  D2_jac = jac_finite_diff(t_new, lambda t: g_norm(t)[1], n=2)
 
  ## Subgradient of  g^*(y)
  usv1 = np.linalg.svd(D1.A, compute_uv=True, full_matrices=False) 
  uv1 = usv1[0][:,[0]] @ usv1[2][[0],:]
  grad_d1 = (vec(uv1).T @ vec(D1_jac.A)).item()

  usv2 = np.linalg.svd(D2.A, compute_uv=True, full_matrices=False) 
  uv2 = usv2[0][:,[0]] @ usv2[2][[0],:]
  grad_d2 = (vec(uv2).T @ vec(D2_jac.A)).item()

  grad_d1 + grad_d2

  #t = np.minimum(np.maximum(0.0+h_eps, grad_g), np.max(T))
  t = t_new + np.sign(grad_g)*0.01

  iterates.append([t_new, t, cc])
  cc += 1


## Normalized case - lex order
# D0_jac = jac_finite_diff(t, lambda t: h_norm(t)[0])
# usv0 = np.linalg.svd(h_norm(t)[0].A, full_matrices=False)
# -56.19628354317169
# (vec(usv0[0] @ usv0[2]).T @ vec(D0_jac.A)).item()

# ## Normalized case - ordered 
# D0_jac = D_jac(t, lambda t: (1/m0)*np.flip(np.sort(np.maximum(birth - pdist(C(t)), 0.0))))
# usv0 = np.linalg.svd(np.diag((1/m0)*np.flip(np.sort(np.maximum(birth - pdist(C(t)), 0.0)))), full_matrices=False)
# # (vec(usv0[0] @ usv0[2]).T @ vec(np.diag(D0_jac))).item()
# # -56.19628354317169

# ## Normalized case - unordered 
# D0_jac = D_jac(t, lambda t: (1/m0)*np.maximum(birth - pdist(C(t)), 0.0))
# usv0 = np.linalg.svd(np.diag((1/m0)*np.maximum(birth - pdist(C(t)), 0.0)), full_matrices=False)
# (vec(usv0[0] @ usv0[2]).T @ vec(np.diag(D0_jac))).item()
# -56.19628354317169

#usv0 = np.linalg.svd(np.diag((1/m0)*np.maximum(birth - pdist(C(t)), 0.0)), full_matrices=False)


# usv0 = np.linalg.svd(h_norm(t)[0].A, full_matrices=False)
#usv0 = np.linalg.svd(np.diag(np.maximum(birth - pdist(C(t)), 0.0)), full_matrices=False)
#D0_jac = D_jac(t, lambda t: (1/m0)*np.maximum(birth - pdist(C(t)), 0.0))
#(vec(usv0[0] @ usv0[2]).T @ vec(np.diag(D0_jac))).item()
# usv0 = np.linalg.svd(np.diag(np.flip(np.sort(np.maximum(birth - pdist(C(t)), 0.0)))), full_matrices=False)
#D0_jac = D_jac(t, lambda t: np.flip(np.sort(np.maximum(birth - pdist(C(t)), 0.0))))
#D0_jac = D_jac(t, lambda t: (1/m0)*np.maximum(birth - pdist(C(t)), 0.0))



## Basic DC program: g(x) - h(x)
g = lambda x: 12*x**2
h = lambda x: 20 + x**4

dom = np.linspace(-5, 5, 100)
plt.plot(dom, g(dom), c='blue')
plt.plot(dom, h(dom), c='red')
plt.plot(dom, g(dom)-h(dom), c='black')

# calc_conjugate('12*x**2')
g_conj = lambda y: (1/48)*y**2
g_conj_grad = grad(g_conj)
h_grad = grad(h)


def DC_solve(x0, h_grad: Callable, g_conj_grad: Callable, N: int = 100, keep_iterates: bool = False):
  x, y = x0, h_grad(x0)
  X, Y = [], []
  cc = 0
  while (cc < N):
    y = h_grad(x)
    x = g_conj_grad(y)
    X.append(x), Y.append(y)
    cc += 1 
    if cc > 10 and np.allclose(X[-2], X[-1]):
     break
  return(X[-1] if not(keep_iterates) else (X, Y))

# DC subgradient
def DC_subgradient(x0, h_grad, g_grad, alpha: float = 1e-6, N: int = 100, keep_iterates: bool = False):
  X = []
  x = x0
  for k in range(N):
    uk = h_grad(x)
    #if abs(x)==float('inf'): 
    x = x - alpha*(g_grad(x) - uk)
    X.append(x)
  return(X[-1] if not(keep_iterates) else X)

results = [DC_subgradient(x, h_grad, grad(g), alpha=1e-3, N=1500) for x in dom]

results = [DC_solve(x, h_grad, g_conj_grad, N=100) for x in dom]

for i, r in enumerate(results):
  if np.isinf(r):
    results[i] = np.sign(r)*np.sqrt(np.finfo(float).max)

def argmin_numeric(values):
  values[np.isnan(values)] = np.inf
  return(np.argmin(values))

critical_points = np.unique(np.round(np.array(results), 5))
critical_points = np.array([np.sign(r)*np.finfo(float).max if np.isinf(r) else r for r in critical_points])
crit_ids = np.array([argmin_numeric(abs(r - critical_points)) for r in results])

plt.plot(dom, g(dom)-h(dom), c='black')
plt.scatter(dom, g(dom)-h(dom), c=crit_ids)










## e-approximation to rank
# A = np.array([np.sum(np.diag(X.T @ np.linalg.inv((X @ X.T) + eps*np.eye(X.shape[0])) @ X)) for eps in np.linspace(0.5, 0.0001, 100)]) 
np.linalg.matrix_rank(D3.A)

X = D2.A
A = (X @ X.T) + 0.000000001*np.eye(X.shape[0])
T = np.diag(X.T @ np.linalg.inv(A) @ X)
np.sum(T)

## Boom: approximation to rank iff term is a boundary matrix
np.sum([np.dot(X[:,j], np.linalg.solve(A, X[:,j])) for j in range(X.shape[1])])

Y = X @ X.T
np.all(np.diag(Y) >= 0)  # positive diagonal 
np.allclose(Y, Y.T)      # symmetric
np.all(Y[np.triu_indices(Y.shape[0], k=1)] <= 0.0) # not all entries negative

C, params = circle_family(n=50, sd=0.15)
T = np.linspace(0, 1.5, 1000)
diagrams = [ripser(C(t))['dgms'][1] for t in T]
birth, death = diagrams[int(len(diagrams)/2)][0]*np.array([1.10, 0.90])

D = pdist(C(np.median(T)))
D2, (ew, tw) = rips_boundary(D, p=2, diam=np.inf, sorted=False)
D2.data = np.sign(D2.data)*np.maximum(death - np.repeat(tw, 3), 0.0) 

plt.spy(D2 @ D2.T, markersize=1.0)


def rank_approx1(X: ArrayLike, eps: float = 0.001) -> float:
  assert isinstance(X, np.ndarray) 
  Y = np.linalg.inv((X.T @ X) + eps*np.eye(X.shape[1]))
  #return(np.sum(np.diag(X @ Y @ X.T)))
  return(np.sum(np.diag(Y @ X.T @ X)))
  #return(np.sum(np.diag(np.eye(Y.shape[0]) - Y)))

def rank_approx2(X: ArrayLike, eps: float = 0.001) -> float:
  assert isinstance(X, np.ndarray) 
  Y = np.linalg.inv((X @ X.T) + eps*np.eye(X.shape[0]))
  return(np.sum(np.diag(X.T @ Y @ X)))

X = D2.A
rank_approx1(X)
rank_approx2(X)

def run_chol():
  M = X @ X.T
  factor = cholesky(csc_matrix(M + 0.001*np.eye(M.shape[0])))
  return(np.sum([factor(M[:,j])[j] for j in range(M.shape[1])]))

timeit.timeit(run_chol, number=100)
timeit.timeit(lambda: rank_approx1(X), number=10)
timeit.timeit(lambda: rank_approx2(X), number=10)
## Is the fourth term still a boundary matrix?? Well it's an incidence matrix?

from sksparse.cholmod import analyze, cholesky, analyze_AAt, cholesky_AAt
cholesky(csc_matrix(M), beta=0.001)


factor = analyze(csc_matrix(M), ordering_method="best")
## It works!
M_sparse = csc_matrix(M)
timeit.timeit(lambda: factor.cholesky_inplace(M_sparse, beta=0.001), number=100)


X = D2.A
M = X @ X.T
factor = cholesky(csc_matrix(M), beta=0.00001, ordering_method="natural")
plt.spy(factor.L(), markersize=1.0)
factor = analyze(csc_matrix(M), ordering_method="best")
P = factor.P()
factor.cholesky_inplace(csc_matrix(M), beta=0.00001)
plt.spy(factor.L(), markersize=1.0)


factor = cholesky_AAt(D2, beta=0.00001, ordering_method="natural")
plt.spy(factor.L(), markersize=1.0)
factor = analyze_AAt(D2, ordering_method="natural", mode="supernodal")
factor = analyze_AAt(D2, ordering_method="amd", mode="supernodal")
factor = analyze_AAt(D2, ordering_method="metis", mode="supernodal")
factor = analyze_AAt(D2, ordering_method="nesdis", mode="supernodal")
factor = analyze_AAt(D2, ordering_method="colamd", mode="supernodal")
factor = analyze_AAt(D2, ordering_method="best", mode="supernodal")
P = factor.P()
factor.cholesky_AAt_inplace(D2, beta=0.00001)
plt.spy(factor.L(), markersize=1.0)

timeit.timeit(lambda: factor.cholesky_AAt_inplace(D2, beta=0.00001), number=10)


## Approximation
np.sum([factor(M[:,j])[j] for j in range(M.shape[1])])

def run_chol():
  M = X @ X.T
  factor.cholesky_inplace(csc_matrix(M), beta=0.001)
  return(np.sum([factor(M[:,j])[j] for j in range(M.shape[1])]))

import timeit
timeit.timeit(lambda: ripser(C(np.median(T)))['dgms'][1], number=10) # 0.018287795999981427
timeit.timeit(lambda: rank_approx2(X), number=10) #57.01956498500002
timeit.timeit(lambda: np.linalg.matrix_rank(D2.A), number=10) # 23.842623383999978
timeit.timeit(lambda: np.linalg.svd(D2.A, compute_uv=False, full_matrices=False), number=10) # 17.94393631899993
timeit.timeit(run_chol, number=10) #3.4887014009999575
timeit.timeit(lambda: factor.cholesky_AAt_inplace(D2, beta=0.00001), number=10) # 0.09726164599999265


factor.cholesky_AAt_inplace(D2, beta=0.00001)
M = D2 @ D2.T 

timeit.timeit(lambda: np.sum([factor(M[:,j])[j,0] for j in range(M.shape[1])]), number = 10) # 5.745845374000055


import scipy.sparse.linalg
from scipy.linalg.interpolative import estimate_rank
LD2 = scipy.sparse.linalg.aslinearoperator(D2)
np.linalg.matrix_rank(D2.A)

timeit.timeit(lambda: estimate_rank(LD2, 0.0001), number=10)


M_perm = M[P[:, np.newaxis], P[np.newaxis, :]]
factor1 = cholesky(csc_matrix(M_perm), beta=0.00001, ordering_method="natural")
plt.spy(factor1.L())

#M_perm = M[P[:, np.newaxis], P[np.newaxis, :]]
#factr

M_perm = M[P[:, np.newaxis], P[np.newaxis, :]]
factor.cholesky_inplace(csc_matrix(M_perm), beta=0.001)
plt.spy(factor.L())
factor = analyze(csc_matrix(M), ordering_method="best")

def count_dot(x, y): 
  return(len(np.intersect1d(np.flatnonzero(x), np.flatnonzero(y))))
from itertools import product

counts = np.array([count_dot(X[i,:], X[j,:]) for i,j in product(range(X.shape[0]), range(X.shape[0]))])



# upper bound (dumb)
# 16*120*16 = 30720 
# non-zero columns: 16*63*16 = 16128

# actual 
# 228 

# np.sum([np.all(X[:,j] == 0) for j in range(X.shape[1])]) == 63 


# %%




# %% Archimedean circle around sphere


