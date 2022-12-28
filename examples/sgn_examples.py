import numpy as np 
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

x = np.linspace(-1, 1, 1000)
p, eps = 1.0, 0.001
s1 = np.vectorize(lambda x: x**p / (x**p + eps**p))
s2 = np.vectorize(lambda x: x**p / (x**p + eps))
s3 = np.vectorize(lambda x: x / (x**p + eps**p)**(1/p))
s4 = np.vectorize(lambda x: 1 - np.exp(-x/eps))

def sgn_approx(x, eps: float, p: float = 1.0, normalize: bool = False, method="") -> ArrayLike:
  np.interp(0, [0, 1], np.log([1e-8+1, 10+1]))
  pass
# np.interp(0, [0, 1], [np.log(1e-8+1), np.log(10+1)])

fig, axes = plt.subplots(1, 4, figsize=(12, 4))

np.linspace(np.log(1e-8+1), np.log(10+1), 100)

for eps in [1e-12, 1e-8, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1.0, 2, 5, 10, 100]:
  axes[0].plot(x, s1(abs(x)), label=f"eps={eps:.{2}f}")

axes[0].legend()
plt.legend()