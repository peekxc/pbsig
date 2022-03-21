# %% Imports
from dms import circle_family
from betti import *
from persistence import * 
from scipy.spatial.distance import pdist 
import matplotlib.pyplot as plt

# %% Create the family 
C, params = circle_family(n=16, sd=0.15)
T = np.linspace(0, 1.5, 1000)

# %% Get birth and death w/ ripser
from ripser import ripser
from persim import plot_diagrams
diagrams = [ripser(C(t))['dgms'][1] for t in T]
birth, death = diagrams[int(len(diagrams)/2)][0]*np.array([1.10, 0.90])

# %% Get persistent betti number w/ Ripser
PBS = np.array([persistent_betti_rips(C(t), b=birth, d=death, summands=True) for t in T])
PB = PBS[:,0] - (PBS[:,1] + PBS[:,2])
PB_ripser = np.array([np.sum(np.logical_and(dgm[:,0] <= birth, dgm[:,1] > death)) for dgm in diagrams])
# np.all((PB - PB_ripser) == 0)

# %% Show evolution of persistence diagrams 
from persim import plot_diagrams
# cm = plt.get_cmap('jet')
plot_diagrams(diagrams[1], xy_range=[0, 3.0, 0.0, 3.0], labels='H1')
plt.plot(*np.vstack(diagrams).T)
plt.scatter(birth, death, c='red', s=4.5)

# %% Compare relaxation vs. objective vs. Moreau envelope 
m1, m2 = nuclear_constants(params['n_points'], birth, death, 'global')
a1, a2 = 1/m1, 1/m2

ALPHA1 = np.exp(np.linspace(np.log(1), np.log(100), 25))*(a1)
ALPHA2 = np.exp(np.linspace(np.log(1), np.log(100), 25))*(a2)
MU = np.linspace(0.05, 15.0, 25) # random

smooth_pb = np.zeros(shape=(len(T), len(ALPHA1), len(MU), 4))

pb_baseline = np.zeros(shape=(len(T), 4))
for ti,t in enumerate(T):
  D1, D2, D3 = persistent_betti_rips_matrix(C(t), b=birth, d=death)
  USV1 = np.linalg.svd(D1.A, compute_uv=True, full_matrices=False)
  USV2 = np.linalg.svd(D2.A, compute_uv=True, full_matrices=False)
  USV3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False) 
  M0 = np.sum(pdist(C(t)) <= birth)
  tol1 = np.max([np.finfo(float).resolution*100, np.max(D1.shape) * np.finfo(float).eps])
  tol2 = np.max([np.finfo(float).resolution*100, np.max(D2.shape) * np.finfo(float).eps])
  tol3 = np.max([np.finfo(float).resolution*100, np.max(D3.shape) * np.finfo(float).eps])
  pb_baseline[ti,:] = (M0, np.sum(USV1[1] >= tol1), np.sum(USV2[1] >= tol2), np.sum(USV3[1] >= tol3))
pb_baseline = pb_baseline.astype(int)

pb_baseline[:,0] - pb_baseline[:,1] - pb_baseline[:,2] + pb_baseline[:,3]

pb_compare = np.zeros(shape=(len(T), 4))
for ti,t in enumerate(T):
  D1, D2, D3 = persistent_betti_rips_matrix(C(t), b=birth, d=death)
  USV1 = np.linalg.svd(D1.A, compute_uv=True, full_matrices=False)
  USV2 = np.linalg.svd(D2.A, compute_uv=True, full_matrices=False)
  USV3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False) 
  M0 = np.sum(pdist(C(t)) <= birth)
  M1, prox_af1 = prox_moreau(USV1, alpha=1/m1, mu=2.0)
  M2, prox_af2 = prox_moreau(USV2, alpha=1/m2, mu=2.0) # alpha >= (1/m1) to ensure convexity, mu > 1 to ensure lipshitz constant goes down
  M3, prox_af3 = prox_moreau(USV3, alpha=1/m2, mu=2.0)
  pb_compare[ti,:] = (M0, M1, M2, M3)

K1, K2, K3, K4 = Lipshitz(pb_baseline[:,0], T), Lipshitz(pb_baseline[:,1], T), Lipshitz(pb_baseline[:,2], T), Lipshitz(pb_baseline[:,3], T)
plt.plot(T, pb_compare[:,0] - pb_compare[:,1] - pb_compare[:,2] + pb_compare[:,3])

## Estimate Lipshitz constant of DMS 
DMS_K = 0.0
for i in range(len(T)-1):
  DMS_K = np.max([DMS_K, np.max(abs(pdist(C(T[i]))-pdist(C(T[i+1])))/abs(T[i+1]-T[i]))])

## Get rank of D1 
D1_r = np.zeros(len(T))
for i in range(len(T)):
  D1, D2, D3 = persistent_betti_rips_matrix(C(T[i]), b=birth, d=death)
  D1_r[i] = np.linalg.matrix_rank(D1.A)
D1R_K = Lipshitz(D1_r, T)

## Estimate Lipshitz constant of D1
D1_K = 0.0
D1_s = np.zeros(len(T))
for i in range(len(T)):
  D1, D2, D3 = persistent_betti_rips_matrix(C(T[i]), b=birth, d=death)
  USV1 = np.linalg.svd((1/m1)*D1.A, compute_uv=True, full_matrices=False)
  D1_s[i] = np.sum(USV1[1])
D1_K = Lipshitz(D1_s, T)

## Estimate Lipshitz constant for Moreau
D1_m = np.zeros(len(T))
for i in range(len(T)):
  D1, D2, D3 = persistent_betti_rips_matrix(C(T[i]), b=birth, d=death)
  USV1 = np.linalg.svd((1/m1)*D1.A, compute_uv=True, full_matrices=False)
  M1, prox_af1 = prox_moreau(USV1, alpha=1/m1, mu=2.0)
  D1_m[i] = M1
D1M_K = Lipshitz(D1_m, T)
  
for i in range(len(T)-1):
  DMS_K = np.max([DMS_K, np.max(abs(pdist(C(T[i]))-pdist(C(T[i+1])))/abs(T[i+1]-T[i]))])

for ti,t in enumerate(T):
  D1, D2, D3 = persistent_betti_rips_matrix(C(t), b=birth, d=death)
  USV1 = np.linalg.svd(D1.A, compute_uv=True, full_matrices=False)
  USV2 = np.linalg.svd(D2.A, compute_uv=True, full_matrices=False)
  USV3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False) 
  M0 = np.sum(pdist(C(t)) <= birth)
  for i, (A1, A2) in enumerate(zip(ALPHA1, ALPHA2)):
    for j, mu in enumerate(MU): 
      M1, prox_af1 = prox_moreau(USV1, alpha=A1, mu=mu)
      M2, prox_af2 = prox_moreau(USV2, alpha=A2, mu=mu)
      M3, prox_af3 = prox_moreau(USV3, alpha=A2, mu=mu)
      smooth_pb[ti,i,j,:] = (M0,M1,M2,M3)
  # smooth_pb[i,:] = (M0,M1,M2,M3)



def pb_relax2(X, b, d, a1, a2, mu):
  D1, D2, D3 = persistent_betti_rips_matrix(C(t), b=b, d=d)
  USV1 = np.linalg.svd(D1.A, compute_uv=True, full_matrices=False)
  USV2 = np.linalg.svd(D2.A, compute_uv=True, full_matrices=False)
  USV3 = np.linalg.svd(D3.A, compute_uv=True, full_matrices=False) 
  M0 = np.sum(pdist(C(t)) <= b)
  # M1, prox_af1 = prox_moreau(USV1, alpha=a1, mu=mu)
  # M2, prox_af2 = prox_moreau(USV2, alpha=a2, mu=mu)
  # M3, prox_af3 = prox_moreau(USV3, alpha=a2, mu=mu)
  M1 = a1*np.sum(USV1[1])
  M2 = a2*np.sum(USV2[1])
  M3 = a2*np.sum(USV3[1])
  return(M0, M1, M2, M3)


## Remark: we should only inherit smoothness properties when we know the singular values are <= 1
baseline = np.array([relaxed_pb(C(t), b=birth, d=death, bp=birth, m1=m1, m2=m2) for t in T])



relaxed_pb(C(0.5), b=birth, d=death, bp=birth, m1=m1, m2=m2, summands=True)
pb_relax2(C(0.5), birth, death, 1/m1, 1/m2, 1.0)

## Fixing alpha, varying mu  
plt.plot(T, PB, label='PB')
plt.plot(T, baseline, label='baseline')
ai = 10
for j in range(10, 25):# range(len(MU)):
  A = smooth_pb[:,ai,j,:]
  Fa = A[:,0] - A[:,1] - A[:,2] + A[:,3]
  plt.plot(T, Fa, label=f"{MU[j]:.2f}")
  print(Lipshitz(Fa, T))
plt.legend(fontsize=8, title_fontsize=15)

## Look at lipshitz constants for terms
ai = 20
L = np.zeros(shape=(len(MU), 3))
for j in range(len(MU)):
  A = smooth_pb[:,ai,j,:]
  L[j,:] = (Lipshitz(A[:,1], T), Lipshitz(A[:,2], T), Lipshitz(A[:,3], T))

plt.plot(L[:,0])
plt.plot(L[:,1])
plt.plot(L[:,2])

# Check functions are monotonically more smooth (smaller Lipshitz constants) w/ additional smoothing 
np.all(np.ediff1d(L[:,0]) <= 0), np.all(np.ediff1d(L[:,1]) <= 0), np.all(np.ediff1d(L[:,2]) <= 0)


## Fixing mu, varying alpha  
plt.plot(T, PB)
mj = 10
for i in range(len(ALPHA1)):
  A = smooth_pb[:,i,mj,:]
  plt.plot(T, A[:,0] - (A[:,1] + A[:,2] + A[:,3]))


## Individual term plots 
from tallem.color import bin_color
from matplotlib import cm
mu_col = bin_color(MU, cm.turbo.colors)
fig, axs = plt.subplots(1, 4, figsize=(18, 3))
ai = 10
for j in range(len(MU)):
  A = smooth_pb[:,ai,j,:]
  axs[0].plot(T, A[:,0], c=mu_col[j])
  axs[1].plot(T, A[:,1], c=mu_col[j])
  axs[2].plot(T, A[:,2], c=mu_col[j])
  axs[3].plot(T, A[:,3], c=mu_col[j])
axs[1].set_yscale("log")
axs[2].set_yscale("log")
axs[3].set_yscale("log")


# %% Enumerate the singular values for T1, T2 for the whole family
def singular_values_T1(t):
  D1, (vw, ew) = rips_boundary(C(t), p=1, diam=death, sorted=False)
  D1.data = np.sign(D1.data) * np.maximum(birth - np.repeat(ew, 2), 0.0)
  return(np.linalg.svd(D1.A, compute_uv=False))

def singular_values_T2(t, method='anderson'):
  D1, (vw, ew) = rips_boundary(C(t), p=1, diam=death, sorted=True)
  D1.data = np.sign(D1.data) * np.maximum(birth - np.repeat(ew, 2), 0.0)
  D2, (ew, tw) = rips_boundary(C(t), p=2, diam=death, sorted=True)
  D2.data = np.sign(D2.data) * np.maximum(death - np.repeat(tw, 3), 0.0)
  Z, B = D1[:,ew <= birth], D2[ew <= birth,:]
  PI = projector_intersection(Z, B, space='NR', method=method)
  spanning_set = PI @ B
  return(np.linalg.svd(spanning_set, compute_uv=False))

S1 = [singular_values_T1(t) for t in T]
S2 = [singular_values_T2(t) for t in T]

# %% Plot Moreau envelopes of L1 norm 
dom = np.linspace(0, 1, 1000)
for h in np.linspace(0, 1, 20):
  huber = Huber(h)
  plt.plot(dom, huber(dom), linestyle='dashed' if h == 0 else 'solid')
# plt.plot(T, np.array([np.sum(s) for s in S1]))
# plt.scatter(T, np.array([np.sum(s) for s in S1]), s=1.5)

# %% For S 
for h in np.linspace(0, 1, 20):
  huber = Huber(h)
  plt.plot(dom, 1.0-huber(dom), linestyle='dashed' if h == 0 else 'solid')

# %% S1
for h in np.linspace(0, 1, 20):
  huber = Huber(h)
  plt.plot(T, np.array([np.sum(huber(s)) for s in S1]))

# %% Estimate Lipshitz constant for Huber-smoothing
K_S1 = np.zeros(200)
m1 = np.max(singular_values_T1(0))
for i, h in enumerate(np.linspace(0, 1, 200)):
  huber = Huber(h)
  s1_smoothed = np.array([np.sum(huber(s/m1)) for s in S1])
  K_S1[i] = np.max(np.abs(np.diff(s1_smoothed))/np.abs(np.diff(T)))

# %% 
K_S2 = np.zeros(200)
m2 = np.max(singular_values_T2(0))
for i, h in enumerate(np.linspace(0, 1, 200)):
  huber = Huber(h)
  s2_smoothed = np.array([np.sum(huber(s/m2)) for s in S2])
  K_S2[i] = np.max(np.abs(np.diff(s2_smoothed))/np.abs(np.diff(T)))

# %%
plt.plot(np.linspace(0, 1, 200), K_S1, label='T1 Lipshitz Constant (Huber)') 
plt.plot(np.linspace(0, 1, 200), K_S2, label='T2 Lipshitz Constant (Huber)')

## Plot the normalized objective function under various Huber settings + power settings


# %% 
m1 = np.max(singular_values_T1(0))
H = np.linspace(0, 1, 25)
PBR = np.zeros(shape=(len(H), len(T), 3))
for i, h in enumerate(H):
  PBR[i,:] = np.array([relaxed_pb(C(t),b=1.0, d=1.1, bp=1.10, m1=m1, m2=m2, summands=True, huber=h) for t in T])
  print(i)

# %% Plot the objective under different smoothing  
for i in range(len(H)): 
  plt.plot(T, PBR[i,:,0] - (PBR[i,:,1] + PBR[i,:,2]))

# %% 
for i in range(len(H)): 
  plt.plot(T, PBR[i,:,2])


# %% Estimate Lipshitz constant of DMS 
K = 0.0
for i in range(len(T)-1):
  diff_f = np.max(np.abs(pdist(C(T[i]))-pdist(C(T[i+1]))))
  diff_d = np.abs(T[i+1]-T[i])
  K = np.max([K, diff_f/diff_d])

# %% Line-search for prox function
t = np.median(T)
D1, (vw, ew) = rips_boundary(C(t), p=1, diam=death, sorted=False)
D1.data = np.sign(D1.data) * np.maximum(birth - np.repeat(ew, 2), 0.0)

def prox(X, U, mu, f):
  """ Proximal function for sparse matrices X and U """
  f_u = f(U)
  p2 = (1/(2*mu))*np.linalg.norm((X - U).A)**2
  return(f_u + p2)

fU = lambda U: (1/m1)*np.sum(np.linalg.svd(U.A, compute_uv=False))

ls = np.zeros(100)
for i, c in enumerate(np.linspace(1.0, 5.25, 100)):
  D1_c = D1.copy()
  D1_c.data = (1/c)*(np.sign(D1_c.data) * np.maximum(birth - np.repeat(ew, 2), 0.0))
  ls[i] = prox(D1, D1_c, 0.50, fU)
  
plt.plot(np.linspace(1.0, 5.25, 100), ls)

c_min = np.linspace(1.0, 5.25, 100)[np.argmin(ls)]
D1_c = D1.copy()
D1_c.data = (1/c_min)*(np.sign(D1_c.data) * np.maximum(birth - np.repeat(ew, 2), 0.0))

np.sum(np.linalg.svd(D1_c.A, compute_uv=False))
np.sum(Huber(0.50)(np.linalg.svd(D1.A, compute_uv=False)))

# %% 
plt.plot(T, pbr[:,1])
plt.plot(T, pbr_h[:,1])
plt.scatter(T, pbr[:,0], s=1.5)
plt.scatter(T, pbr_h[:,0], s=1.5, c='black')


np.max(pdist(pbr[:,[1]])/pdist(np.matrix(T).T))
np.max(pdist(pbr_h[:,[1]])/pdist(np.matrix(T).T))




# %% Solve convex function for T1 
S1 = np.zeros(shape=(params['n_points'], len(T)))
for i, t in enumerate(T):
  sv =  singular_values_T1(t)
  S1[:len(sv),i] = sv
m1 = np.max(np.max(S1, axis=0))
T1 = np.sum(S1, axis=0)

## This should be convex in [min(T), max(T)]!
plt.plot(T, (1/m1)*T1)
plt.scatter(T, (1/m1)*T1, s=1.5, c='black')

## Not strictly needed, but good to check 
np.all(T1[0:(len(T1)-1)] - T1[1:len(T1)] >= 0)


# %% Solve convex function for T2 
S2 = np.zeros(shape=(int((params['n_points']*(params['n_points']-1))/2), len(T)))
for i, t in enumerate(T):
  sv =  singular_values_T2(t)
  S2[:len(sv),i] = sv
m2 = np.max(np.max(S2, axis=0))
T2 = np.sum(S2, axis=0)

## This might not be convex in [min(T), max(T)]!
plt.plot(T, (1/m2)*T2)
plt.scatter(T, (1/m2)*T2, s=1.5, c='black')

SV = np.zeros(shape=(len(T), 4))
SV2 = np.zeros(shape=(len(T), 3))
for i,t in enumerate(T):
  T1, T2_a, T2_b = persistent_betti_rips_matrix(C(t), birth, death, 1)
  SV[i,:] = np.sum(pdist(C(t)) <= birth), nuclear_norm(T1.A), nuclear_norm(T2_a.A), nuclear_norm(T2_b.A)
  SV2[i,:] = relaxed_pb(C(t), birth, death, birth, m1, m2, True)

plt.plot(T, SV[:,0] - (1/m1)*SV[:,1] - (1/m2)*(SV[:,2] - SV[:,3]))
plt.plot(T, SV2[:,0] - (SV2[:,1] + SV2[:,2]))
np.all(T2[0:(len(T2)-1)] - T2[1:len(T2)] >= 0)



# %% Relaxed objective 
m1, m2 = nuclear_constants(params['n_points'], birth, death, "tight", C, T)

P = np.exp(np.linspace(np.log(1), np.log(11), 20))-1.0
# P = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0])
T1 = np.zeros((len(T), len(P)))
for j, p in enumerate(P): 
  for i, t in enumerate(T):
    T1[i,j] = relaxed_pb(C(t), birth, death, birth, m1, m2, summands=True, proximal=p)[1]

# %% 
for j in range(T1.shape[1]):
  plt.plot(T, T1[:,j], label=f"p={P[j]:.2f}")
plt.legend()

# %%
T1 = [relaxed_pb(C(t), birth, death, birth, m1, m2, summands=False, proximal=2.05) for t in T]
plt.plot(T, T1)

# %% 
TS = np.array([relaxed_pb(C(t), birth, death, birth, m1, m2, summands=True, proximal=3.5) for t in T])

plt.plot(T, TS[:,1])

# %%
sv = np.zeros(shape=len(T))
for i,t in enumerate(T):
  DA, DB, DC = persistent_betti_rips_matrix(C(t), b=birth,d=death)
  sv[i] = np.sum(np.linalg.svd(DC.A, compute_uv=False))

plt.plot(T, sv)




# %% vineyards 

reduction_pHcol()
