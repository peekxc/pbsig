import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('/Users/mpiekenbrock/Downloads/dog7.mat')

x = np.ravel(mat['surface']['X'][0][0]).astype(float)
y = np.ravel(mat['surface']['Y'][0][0]).astype(float)
z = np.ravel(mat['surface']['Z'][0][0]).astype(float)
S = np.c_[x,y,z]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(*S.T, s=0.15)
lb,ub = min(S.min(axis=0)), max(S.max(axis=0))
ax.set_xlim(lb, ub)
ax.set_ylim(lb, ub)
ax.set_zlim(lb, ub)