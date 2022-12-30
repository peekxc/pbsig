import numpy as np 
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

x = np.linspace(-1, 1, 1000)
p, eps = 1.0, 0.001
s1 = np.vectorize(lambda x: x**p / (x**p + eps**p))
s2 = np.vectorize(lambda x: x**p / (x**p + eps))
s3 = np.vectorize(lambda x: x / (x**p + eps**p)**(1/p))
s4 = np.vectorize(lambda x: 1 - np.exp(-x/eps))

## NEED TO FIGURE OUT QUANTILE NORMALIZATION

def sgn_approx(x: Optional[ArrayLike] = None, eps: float = 0.0, p: float = 1.0, normalize: bool = False, method: int = 0) -> Union[ArrayLike, Callable]:
  """ 
  Parameters: 
    method := integer in { 0, 1, 2, 3 } 
  """
  # from pbsig.color import scale_interval
  assert method >= 0 and method <= 3, f"Invalid method {method} chosen. Must be between [0,3]"
  # eps = scale_interval([eps], scaling="logarithmic", min_x=0, max_x=10)[0] if normalize else eps
  if normalize:
    assert eps >= 0.0 and eps <= 1.0, "If normalize = True, eps must be in [0,1]"
    dom = np.linspace(0, 1, 100)
    EPS = np.linspace(1e-16, 10, 150) # arbitrary
    A = np.array([np.trapz(sgn_approx(eps, p, normalize=False, method=method)(dom), dom) for eps in EPS])
    A[0] = 1
    eps = np.interp(x=np.quantile(A, eps), xp=np.cumsum(A)/np.sum(A), fp=EPS)
    return sgn_approx(eps, p, normalize=False, method=method)
  else:
    f1 = np.vectorize(lambda x: x**p / (x**p + eps**p))
    f2 = np.vectorize(lambda x: x**p / (x**p + eps))
    f3 = np.vectorize(lambda x: x / (x**p + eps**p)**(1/p))
    f4 = np.vectorize(lambda x: 1 - np.exp(-x/eps))
    f = [f1,f2,f3,f4][method]
    def _sgn_approx(x: ArrayLike) -> ArrayLike:
      return f(x)
    return np.vectorize(_sgn_approx)

# np.interp(0, [0, 1], [np.log(1e-8+1), np.log(10+1)])

## Normalized plots
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
x = np.linspace(-1,1,1000)
for eps in np.linspace(0, 1, 8):
  for method in range(4):
    S = sgn_approx(eps=eps, p=2, normalize=True, method=method)
    axes[method].plot(x, S(abs(x)), label=f"eps={eps:.{2}f}")


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