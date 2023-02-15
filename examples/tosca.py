import scipy.io
import matplotlib.pyplot as plt
import numpy as np

mat = scipy.io.loadmat('/Users/mpiekenbrock/Downloads/dog7.mat')

x = np.ravel(mat['surface']['X'][0][0]).astype(float)
y = np.ravel(mat['surface']['Y'][0][0]).astype(float)
z = np.ravel(mat['surface']['Z'][0][0]).astype(float)
S = np.c_[x,y,z]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(*S.T, s=0.15, c=S[:,2])
c = S.mean(axis=0)
rng = max(abs(S.max(axis=0)-S.min(axis=0)))
ax.set_xlim(c[0] - 0.5*rng, c[0] + 0.5*rng)
ax.set_ylim(c[1] - 0.5*rng, c[1] + 0.5*rng)
ax.set_zlim(c[2] - 0.5*rng, c[2] + 0.5*rng)



