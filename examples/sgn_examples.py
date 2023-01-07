import numpy as np 
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from pbsig.betti import sgn_approx

# x = np.linspace(-1, 1, 1000)
# p, eps = 1.0, 0.001
# s1 = np.vectorize(lambda x: x**p / (x**p + eps**p))
# s2 = np.vectorize(lambda x: x**p / (x**p + eps))
# s3 = np.vectorize(lambda x: x / (x**p + eps**p)**(1/p))
# s4 = np.vectorize(lambda x: 1 - np.exp(-x/eps))

## NEED TO FIGURE OUT QUANTILE NORMALIZATION

# def sgn_approx(x: Optional[ArrayLike] = None, eps: float = 0.0, p: float = 1.0, normalize: bool = False, method: int = 0) -> Union[ArrayLike, Callable]:
#   """ 
#   Parameters: 
#     method := integer in { 0, 1, 2, 3 } 
#   """
#   # from pbsig.color import scale_interval
#   assert method >= 0 and method <= 3, f"Invalid method {method} chosen. Must be between [0,3]"
#   # eps = scale_interval([eps], scaling="logarithmic", min_x=0, max_x=10)[0] if normalize else eps
#   if normalize:
#     assert eps >= 0.0 and eps <= 1.0, "If normalize = True, eps must be in [0,1]"
#     dom = np.linspace(0, 1, 100)
#     EPS = np.linspace(1e-16, 10, 150) # arbitrary
#     A = np.array([np.trapz(sgn_approx(eps, p, normalize=False, method=method)(dom), dom) for eps in EPS])
#     A[0] = 1
#     eps = np.interp(x=np.quantile(A, eps), xp=np.cumsum(A)/np.sum(A), fp=EPS)
#     return sgn_approx(eps, p, normalize=False, method=method)
#   else:
#     f1 = np.vectorize(lambda x: x**p / (x**p + eps**p))
#     f2 = np.vectorize(lambda x: x**p / (x**p + eps))
#     f3 = np.vectorize(lambda x: x / (x**p + eps**p)**(1/p))
#     f4 = np.vectorize(lambda x: 1 - np.exp(-x/eps))
#     f = [f1,f2,f3,f4][method]
#     def _sgn_approx(x: ArrayLike) -> ArrayLike:
#       return f(x)
#     return np.vectorize(_sgn_approx)

# np.interp(0, [0, 1], [np.log(1e-8+1), np.log(10+1)])

## Normalized plots
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
x = np.linspace(-1,1,1000)
for eps in np.linspace(0, 1, 8):
  for method in range(4):
    S = sgn_approx(abs(x), eps=eps, p=1.0, method=method)
    axes[method].plot(x, S, label=f"eps={eps:.{2}f}")

def _normalize_sgn(x: ArrayLike, alpha: float, p: float = 1.0, method: int = 0):
  from scipy.interpolate import interp1d
  assert alpha >= 0.0 and alpha <= 1.0,"Must be in [0,1]"
  dom = np.abs(np.linspace(0, 1, 100))
  EPS = np.linspace(1e-16, 10, 50)
  area = np.array([2*np.trapz(sgn_approx(dom, eps, p, method), dom) for eps in EPS])
  area[0] = 2.0
  area[-1] = 0.0
  #f = np.interp(alpha, xp=2-area, fp=EPS)
  n_eps = interp1d(2-area, EPS)(alpha*2)
  return sgn_approx(x, n_eps, p, method)

from pbsig.betti import sgn_approx
# S = sgn_approx(normalize=True)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
x = np.linspace(-1,1,1000)
for a in np.linspace(0, 1, 8):
  for method in range(4):
    S = sgn_approx(abs(x), eps=a, p=1.0, method=method, normalize=False)
    #S = _normalize_sgn(abs(x), alpha=a, p=2.0, method=method)
    axes[method].plot(x, S, label=f"eps={a:.{2}f}")
axes[0].legend()

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
x = np.linspace(-1,1,1000)
for a in np.linspace(0, 1, 8):
  for method in range(4):
    S = sgn_approx(eps=a, p=2.0, method=method, normalize=True)
    #S = sgn_approx(abs(x), eps=a, p=2.0, method=method, normalize=True)
    #S = _normalize_sgn(abs(x), alpha=a, p=2.0, method=method)
    axes[method].plot(x, S(abs(x)), label=f"eps={eps:.{2}f}")

plt.plot(area)
np.quantile(area, 0.50)


# 2*np.trapz(sgn_approx(dom, 0.17679068570888795, p, method), dom)

np.quantile(2-area, 0.01)
2.0-np.interp(0.05, EPS, area)


np.trapz(sgn_approx(dom, 0.18731490517708294, 1, 0), dom)

np.quantile(area, 0.50) # what eps yields 0.50 area

np.interp1d

plt.plot(2.0-area)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
np.linspace(np.log(1e-8+1), np.log(10+1), 100)

for eps in [1e-12, 1e-8, 1e-6, 1e-2, 0.05, 1e-1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5, 10, 100]:
  axes[0].plot(x, sgn_approx(abs(x), eps=eps),label=f"eps={eps:.{2}f}")

cuts = np.array([1e-12, 1e-8, 1e-6, 1e-2, 0.05, 1e-1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5, 10, 100])
# scale_interval(, "equalize")

axes[0].legend()
# plt.legend()

## Normalize by function area in [0,1]
p = 2.0
dom = np.linspace(0, 1, 1000)
EPS = np.linspace(1e-16, 10, 150)
A = np.array([np.trapz(np.vectorize(lambda x: x**p / (x**p + eps**p))(dom), dom) for eps in EPS])
A[0] = 1

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for alpha in np.linspace(0, 1, 10):
  eps = np.interp(x=np.quantile(A, alpha), xp=np.cumsum(A)/np.sum(A), fp=EPS)
  f = np.vectorize(lambda x: x**p / (x**p + eps**p))
  axes[0].plot(x, f(abs(x)),label=f"eps={eps:.{2}f}")


np.quantile(A, 0.)

np.quantile(A, q = 0.45)
# cuts = 1.0-scale_interval(A, "equalize")
# plt.plot(A)
# plt.plot(cuts)