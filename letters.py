
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from numpy.typing import ArrayLike
import string
import matplotlib.pyplot as plt 
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

import betti

def expand_triangles(nv: int, E: ArrayLike):
  from gudhi import SimplexTree
  st = SimplexTree()
  for v in range(nv): st.insert([int(v)], 0.0)
  for e in E: st.insert(e)
  st.expansion(2)
  triangles = np.array([s[0] for s in st.get_simplices() if len(s[0]) == 3])
  return(triangles)

LETTERS = list(string.ascii_uppercase)

def gen_letter(text, font_path):
  nwide = 51
  image = Image.new("RGBA", (nwide,nwide), (255,255,255))
  draw = ImageDraw.Draw(image)
  path = Path(font_path).resolve()
  font = ImageFont.truetype(str(path), 40)

  #w,h = draw.textsize(text, font=font)
  w,h = draw.textbbox(xy=(0,0), text=text, font=font)[2:4]
  wo = int((nwide-w)/2)
  ho = int((nwide-h)/2)-int(11/2)
  draw.text((wo, ho), text, fill="black", font=font)

  #image.save("A.gif")

  pixels = np.asarray(image.convert('L'))
  v_pos = np.column_stack(np.where(pixels < 200))
  v_pos = np.fliplr(v_pos)
  v_pos[:,1] = nwide-v_pos[:,1]
  e_ind = np.where(pdist(v_pos, 'chebychev') == 1)[0]
  from apparent_pairs import unrank_C2
  def is_freundenthal(e):
    i,j = unrank_C2(e, v_pos.shape[0])
    x1, x2 = v_pos[i,0], v_pos[j,0]
    y1, y2 = v_pos[i,1], v_pos[j,1]
    return(((x1 < x2) and (y1 == y2)) or ((x1 == x2) and (y1 < y2)) or ((x1 < x2) and (y1 > y2)) or ((x1 == x2) and (y1 > y2)))
  e_fr = np.array(list(filter(is_freundenthal, e_ind)))
  E = np.array([unrank_C2(e, v_pos.shape[0]) for e in e_fr])
  T = expand_triangles(v_pos.shape[0], E)
  center = np.array([
    np.min(v_pos[:,0]) + (np.max(v_pos[:,0])-np.min(v_pos[:,0]))/2,
    np.min(v_pos[:,1]) + (np.max(v_pos[:,1])-np.min(v_pos[:,1]))/2
  ])
  def scale_diameter(X):
    vnorms = np.linalg.norm(X, axis=1)
    vnorms = np.repeat(np.max(vnorms), X.shape[0])
    X = X / np.reshape(np.repeat(vnorms, X.shape[1]), X.shape)
    return(X)
  V = scale_diameter(v_pos - center) 
  return(V, T)

import matplotlib.pyplot as plt
from betti import *
fonts = ['Lato-Bold.ttf', 'Oswald-Bold.ttf', 'OpenSans-Bold.ttf', 'Roboto-Bold.ttf']

V, T = gen_letter('A', 'data/' + fonts[3])
E = edges_from_triangles(T, V.shape[0])
plt.scatter(V[:,0], V[:,1], s=4.30, c='black')
plt.gca().set_aspect('equal')
for e in E: plt.plot(V[e,0], V[e,1], c='black', linewidth=0.80)
for t in T: plt.gca().fill(V[t,0], V[t,1], c='#c8c8c8b3')
plt.gca().set_xlim(-1, 1)
plt.gca().set_ylim(-1, 1)


def lower_star_ph_dionysus(V, W, T):
  # from betti import edges_from_triangles
  E = edges_from_triangles(T, V.shape[0])
  vertices = [([i], w) for i,w in enumerate(W)]
  edges = [(list(e), np.max(W[e])) for e in E]
  triangles = [(list(t), np.max(W[t])) for t in T]
  F = []
  for v in vertices: F.append(v)
  for e in edges: F.append(e) 
  for t in triangles: F.append(t)

  import dionysus as d
  f = d.Filtration()
  for vertices, time in F: f.append(d.Simplex(vertices, time))
  f.sort()
  m = d.homology_persistence(f)
  dgms = d.init_diagrams(m, f)
  DGM0 = np.array([[pt.birth, pt.death] for pt in dgms[0]])
  DGM1 = np.array([[pt.birth, pt.death] for pt in dgms[1]])
  return([DGM0, DGM1])

project_v = lambda V, v: (V @ np.array(v)[:,np.newaxis]).flatten()
W = project_v(V, [1/np.sqrt(2), 1/np.sqrt(2)])


from betti import lower_star_pb
theta = np.linspace(0, 2*np.pi, 124)
C = np.c_[np.cos(theta), np.sin(theta)]

res = {}
for f in fonts:
  for sym in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    V, T = gen_letter(sym, 'data/' + f)
    DGMS = [lower_star_ph_dionysus(V, project_v(V, v), T) for v in C]
    for i, dgms in enumerate(DGMS):
      dgms[0][:,1][dgms[0][:,1] == np.inf] = 2.0
      if dgms[1].ndim > 1:
        dgms[1][:,1][dgms[1][:,1] == np.inf] = 2.0
      DGMS[i] = dgms
    res[(sym, f)] = DGMS

dgm0_dist = np.array([persim.bottleneck(dgms_a[0], dgms_b[0]) for dgms_a, dgms_b in zip(DGMS1, DGMS2)])

from functools import partial
def reorder_from_idx(idx, a):
  return a[idx:] + a[:idx]

def cyclic_perm(a):
  for i in range(len(a)):
    yield partial(reorder_from_idx, i)(a)

def min_bn_dist(dgms_a, dgms_b):
  assert len(dgms_a) == len(dgms_b)
  import persim
  #bn_dist = lambda D1, D2: np.sum([persim.bottleneck(d1, d2) for d1, d2 in zip(D1, D2)])
  bn_dist = lambda D1, D2: np.sum([wasserstein_distance(d1, d2) for d1, d2 in zip(D1, D2)])
  total_bn = np.min([bn_dist(dgms_a, dgms_bp) for dgms_bp in cyclic_perm(dgms_b)])
  return(total_bn)


from itertools import combinations
K = np.array(list(res.keys()))
K = K[np.lexsort(np.rot90(K))]
zeroth_dgm = lambda k: [dgm[0] for dgm in res[k]]
zeroth_bn_dist = [min_bn_dist(zeroth_dgm(tuple(ki)), zeroth_dgm(tuple(kj))) for ki, kj in combinations(K, 2)]


from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
D0 = squareform(zeroth_bn_dist)
# [np.ix_(ind, ind)]
ind = [i for i in range(D0.shape[0]) if i != 9]
#plt.imshow(D0, cmap='viridis')
plt.imshow(D0[np.ix_(ind, ind)], cmap='viridis')
plt.show()

from tallem.dimred import cmds

Z = cmds(D0[np.ix_(ind, ind)])
plt.scatter(*Z.T)
S = np.array([s for s,f in K])[ind]
for i, s in enumerate(S):
  plt.text(Z[i,0], Z[i,1], s=s)



K = np.array(K) 
ind = np.lexsort(np.rot90(K))
K[]

## Wasserstein-1 distance 
import ot
from gudhi.wasserstein import wasserstein_distance
W1 = lambda D1, D2: np.sum([wasserstein_distance(dgm0, dgm1) for dgm0, dgm1 in zip(D1, D2)])
D1 = lower_star_ph_dionysus(V, project_v(V, [0, 1]), T)
D2 = lower_star_ph_dionysus(V, project_v(V, [1, 0]), T)
W1(D1, D2)


## Perturbation expansion 
_, D1_0, _, _, _ = lower_star_pb_terms(V, T, project_v(V, C[0,:]), -0.20, 0.20)
_, D1_1, _, _, _ = lower_star_pb_terms(V, T, project_v(V, C[1,:]), -0.20, 0.20)

U,s,Vt = np.linalg.svd(D1_0.A, compute_uv=True, full_matrices=False)

## This should be small (< 1/10)
np.linalg.norm(E_01.A, 'fro')/np.max(abs(np.ediff1d(S)))

S = np.linalg.svd(D1_0.A, compute_uv=False)
E_01 = D1_1 - D1_0 ## TODO: solve for when thsi frobenius norm is <= some constant based on angle/v
sigma1 = U[:,0].T @ (D1_1 - D1_0) @ (Vt.T)[:,0]
max_error = np.max(np.linalg.svd(E_01.A, compute_uv=False))
new_sigma1 = np.linalg.svd(D1_1.A, compute_uv=False)[0]
print(f"Original: {S[0]} => {S[0] + sigma1} <= {new_sigma1} <= {S[0] + sigma1 + max_error}")
sigma1-max_error



sv1 = np.linalg.svd(D1.A, compute_uv=False)

from betti import lower_star_pb
theta = np.linspace(0, 2*np.pi, 32)
C = np.c_[np.cos(theta), np.sin(theta)]

LETTERS_local = LETTERS[:5]
SIG = np.zeros(shape=(len(LETTERS_local), len(fonts), len(C)))
for j, f in enumerate(fonts):
  for i, text in enumerate(LETTERS_local):
    V, T = gen_letter(text, 'data/'+f)
    SIG[i,j,:] = np.array([lower_star_pb(V, T, project_v(V, v), -0.25, 0.25, type='approx') for v in C])
    print(f"i ({i}), j ({j})")

def cmds(D: ArrayLike, d: int = 2):
  n = D.shape[0]
  H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
  evals, evecs = np.linalg.eigh(-0.5 * H @ D @ H)
  evals, evecs = evals[(n-d):n], evecs[:,(n-d):n]
  w = np.flip(np.maximum(evals, np.repeat(0.0, d)))
  Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
  return(Y)


SIG_COMPARE = []
for i, text in enumerate(LETTERS_local):
  for j, f in enumerate(fonts):
    SIG_COMPARE.append(SIG[i,j,:])
SIG_COMPARE = np.array(SIG_COMPARE)

SIG_EMB = cmds(squareform(pdist(SIG_COMPARE)**2))
plt.scatter(*SIG_EMB.T)
labels = np.repeat(LETTERS_local, len(fonts))
for i, label in enumerate(labels):
  plt.text(SIG_EMB[i,0], SIG_EMB[i,1], label, fontsize=18) 
plt.title(f"a,b = -0.25, +0.25")


np.argsort(A[:,0] - (A[:,1] + A[:,3])) == np.array(range(30))

from itertools import combinations
V, T = gen_letter('A')
grid_ab = np.linspace(0.0, 1.0, 100)
P = np.array([lower_star_pb(V, T, project_v(V, [0, 1]), a, b, type='approx', beta=1e-8) for a,b in combinations(grid_ab, 2)])

XY = np.array([[a,b] for a,b in combinations(grid_ab, 2)])
plt.scatter(*XY.T, c=P)
plt.scatter(dgm1[0][0], 1.0, c='red')


A_sig = np.array([lower_star_pb(V, T, project_v(V, v), 0.40, 0.60, type='approx') for v in C])
B_sig = np.array([lower_star_pb(V, T, project_v(V, v), 0.40, 0.60, type='approx') for v in C])

b1, b2 = 0.10, 0.20
d1, d2 = 0.50, 0.60
pb = lambda b, d, beta: lower_star_pb(V, T, project_v(V, [0,1]) + 1, b, d, type='approx', beta=beta)
pb(b2, d1) - pb(b1, d1) 
pb(b2, d2) - pb(b1, d2) 

from itertools import product
w = np.sort(np.unique(project_v(V, [0,1])))
cc = 0
while cc < 1000:
  b1, b2, d1, d2 = np.random.choice(w, 4)
  eps = 1e-3
  if (b1 <= b2) and (b2 < d1) and (d1 <= d2):
    t1 = pb(b2, d1, eps) - pb(b1, d1, eps) 
    t2 = pb(b2, d2, eps) - pb(b1, d2, eps) 
    assert abs(t1) >= abs(t2) or (abs(t1) - abs(t2)) <= 1e-4
    # np.allclose(abs(t1 - t2), 0.0)
    cc += 1
  else :
    continue


SIG_EMB = cmds(pdist(SIG)**2)
plt.scatter(*SIG_EMB.T)
for i,letter in enumerate(LETTERS):
  plt.text(SIG_EMB[i,0], SIG_EMB[i,1], letter, fontsize=18) 


lower_star_pb(V, T, project_v(V, v), 0.40, 0.60, type='approx') 

dgms = lower_star_ph_dionysus(V, project_v(V, [0, 1]), T)
dgm0 = np.vstack([np.array([p.birth, p.death]) for p in dgms[0]])
dgm1 = np.vstack([np.array([p.birth, p.death]) for p in dgms[1]])


plt.plot(A_sig)
plt.plot(B_sig)
from scipy.sparse import csc_matrix
B = np.array([[1,0,1,0], [-1,1,0,0], [0,-1,-1,-1], [0,0,0,-1]])
#B = B[:,[0,2,3]]
B = csc_matrix(B)
# B[:,2] = 0
B.data = B.data + np.array([0.001, -0.001, 0.002, -0.002, 0.003, -0.003, 0.004, -0.004])
# M = (B @ B.T) + np.eye(B.shape[0])*0.01
#
from sksparse.cholmod import cholesky, cholesky_AAt
#factor = np.linalg.cholesky(M)
factor = cholesky_AAt(B, beta=0.01)
print(factor.L().A)

factor.cholesky_AAt_inplace(B, beta=0.01)
print(factor.L().A)

b = np.array([])
M_base = (D1 @ D1.T)
factor(M_base.A[:,0])

B = np.array([[1,0,1], [-1,1,0], [0,-1,-1, ]])

D = np.diag(1/np.diag(L))
L = (L * D)

Z = np.zeros(size=(4,4))


from gudhi.wasserstein import wasserstein_distance

import matplotlib.pyplot as plt
import numpy as np
from tallem.dimred import cmds
letter_dist = np.array([[ 0. , 10.65742458, 31.44310781, 13.90612661, 35.95402683, 35.54741254, 37.89725145, 34.95980398, 57.97891697, 172.72912344, 67.97973121, 73.81220809, 28.50540287, 29.28713498, 35.07136569, 30.98757885, 59.00229737, 56.12197041, 46.6306051 , 59.1941265 , 44.66173968, 42.89555071, 41.77752184, 43.11120063, 68.32061644, 65.14603125, 65.68641834, 78.55732438], [ 10.65742458, 0. , 29.90799127, 12.00109717, 34.45127098, 34.08206238, 36.0610198 , 33.53731693, 56.22695562, 171.54820577, 66.20763009, 72.21921736, 27.06029257, 28.15202827, 33.42345173, 29.82827174, 57.24051724, 54.23178552, 44.83854602, 57.43040333, 43.66256723, 41.07342553, 39.88392766, 41.84901395, 66.85474201, 63.82797017, 63.77509416, 77.19133046], [ 31.44310781, 29.90799127, 0. , 28.23564209, 30.0722009 , 29.38011093, 25.48656237, 29.1863005 , 57.47934495, 167.86303672, 56.31543797, 67.76280044, 32.30201798, 32.07298182, 23.75809078, 29.84163463, 48.72478256, 44.09602552, 34.43324484, 50.05861466, 38.01670253, 33.55169178, 26.86238671, 36.62259533, 71.81566052, 67.19749864, 53.84269777, 72.20182482], [ 13.90612661, 12.00109717, 28.23564209, 0. , 35.47320277, 35.12352339, 35.99925748, 34.57057841, 56.88069237, 172.69416338, 66.39643062, 72.9113729 , 28.34399795, 29.33009666, 33.77110748, 31.04115862, 57.50106566, 54.43942811, 44.77225936, 57.62475899, 43.07753295, 41.07062472, 39.71176769, 41.51438582, 68.49775088, 65.55193363, 63.789327 , 78.25638756], [ 35.95402683, 34.45127098, 30.0722009 , 35.47320277, 0. , 6.99955123, 13.35435347, 5.15317033, 40.92649518, 154.64538362, 50.03597266, 57.12743435, 17.00064759, 17.83061078, 13.54211524, 15.94647014, 41.53391884, 39.94535473, 32.40097031, 42.02489702, 36.45887744, 33.04496766, 30.7060353 , 34.8030323 , 56.05818707, 51.97626164, 50.11901623, 61.22402847], [ 35.54741254, 34.08206238, 29.38011093, 35.12352339, 6.99955123, 0. , 13.20718075, 5.49790385, 40.91515709, 154.00746871, 49.27877795, 56.68924779, 16.07624388, 17.37120207, 12.73691397, 15.44790733, 40.71316691, 39.11436642, 31.50167232, 40.98442719, 35.81116651, 32.41245948, 30.35891937, 34.20251382, 55.52899446, 51.38360849, 49.70551132, 60.93167753], [ 37.89725145, 36.0610198 , 25.48656237, 35.99925748, 13.35435347, 13.20718075, 0. , 13.32717375, 44.00827605, 155.70242784, 45.70629645, 56.65356509, 19.83120771, 19.98812773, 7.81107998, 17.63853095, 39.56175714, 35.05276887, 25.9536145 , 41.17658427, 34.44990808, 29.47454279, 24.87492583, 32.82868174, 59.69310905, 54.87867452, 44.27126243, 60.6742164 ], [ 34.95980398, 33.53731693, 29.1863005 , 34.57057841, 5.15317033, 5.49790385, 13.32717375, 0. , 40.14338692, 153.34539243, 49.06640753, 56.46178074, 15.6663572 , 16.55145523, 12.35047872, 14.74146824, 40.59836387, 39.04821937, 31.42659777, 40.92228771, 35.4526166 , 32.15452408, 29.82652693, 33.86555348, 54.77179363, 50.687827 , 49.22438041, 60.5146899 ], [ 57.97891697, 56.22695562, 57.47934495, 56.88069237, 40.92649518, 40.91515709, 44.00827605, 40.14338692, 0. , 151.88795991, 45.16476865, 39.38775179, 40.65877496, 43.385273 , 40.73729286, 43.10105603, 44.38095078, 46.66219584, 49.31545741, 41.64553807, 47.70710761, 48.88045288, 54.40050937, 48.27914604, 38.55325502, 45.88218146, 51.01618252, 47.32256434], [172.72912344, 171.54820577, 167.86303672, 172.69416338, 154.64538362, 154.00746871, 155.70242784, 153.34539243, 151.88795991, 0. , 161.63177706, 167.94186768, 152.05066797, 151.66011447, 151.74518968, 149.91691622, 159.56129396, 160.97214311, 162.09566192, 157.8989561 , 159.96856599, 159.47298318, 164.47195627, 160.01193211, 166.1647325 , 160.31935393, 166.61501547, 175.77625682], [ 67.97973121, 66.20763009, 56.31543797, 66.39643062, 50.03597266, 49.27877795, 45.70629645, 49.06640753, 45.16476865, 161.63177706, 0. , 34.27841294, 53.46860516, 53.88028702, 43.972982 , 51.96853453, 45.38413328, 43.92941224, 42.36351092, 44.30355156, 47.17147993, 44.42038965, 47.77566898, 45.78773909, 46.84575567, 50.56998455, 33.47990847, 41.40076211], [ 73.81220809, 72.21921736, 67.76280044, 72.9113729 , 57.12743435, 56.68924779, 56.65356509, 56.46178074, 39.38775179, 167.94186768, 34.27841294, 0. , 57.58484749, 59.49902533, 54.96549824, 59.12884275, 53.27213203, 54.78849677, 56.47060418, 52.11976076, 54.53343724, 54.34704183, 60.8553995 , 53.84452399, 44.92297076, 52.30116582, 51.65090646, 36.93469696], [ 28.50540287, 27.06029257, 32.30201798, 28.34399795, 17.00064759, 16.07624388, 19.83120771, 15.6663572 , 40.65877496, 152.05066797, 53.46860516, 57.58484749, 0. , 9.31248957, 16.0388853 , 10.88842848, 44.72588407, 43.28927735, 35.62533925, 44.87291313, 37.23850825, 34.67224658, 33.20937602, 35.76602818, 51.58668133, 45.78590302, 53.26689779, 62.22876967], [ 29.28713498, 28.15202827, 32.07298182, 29.33009666, 17.83061078, 17.37120207, 19.98812773, 16.55145523, 43.385273 , 151.66011447, 53.88028702, 59.49902533, 9.31248957, 0. , 16.03800084, 8.75789537, 45.27213056, 43.58825567, 35.64040039, 45.36910866, 36.52130938, 33.92944143, 32.66279228, 35.06526933, 54.7645009 , 46.97234601, 53.54782816, 64.20936251], [ 35.07136569, 33.42345173, 23.75809078, 33.77110748, 13.54211524, 12.73691397, 7.81107998, 12.35047872, 40.73729286, 151.74518968, 43.972982 , 54.96549824, 16.0388853 , 16.03800084, 0. , 13.28371679, 37.49340653, 33.01661549, 23.88261331, 39.18968134, 30.91574375, 25.99260308, 22.01089353, 29.35476974, 56.48611768, 51.2714387 , 42.21734349, 58.93869991], [ 30.98757885, 29.82827174, 29.84163463, 31.04115862, 15.94647014, 15.44790733, 17.63853095, 14.74146824, 43.10105603, 149.91691622, 51.96853453, 59.12884275, 10.88842848, 8.75789537, 13.28371679, 0. , 43.3199716 , 41.56030427, 33.26418178, 43.64184861, 34.76399127, 31.73095692, 29.88389113, 33.24355623, 54.09424466, 47.62494125, 51.56686678, 63.62161223], [ 59.00229737, 57.24051724, 48.72478256, 57.50106566, 41.53391884, 40.71316691, 39.56175714, 40.59836387, 44.38095078, 159.56129396, 45.38413328, 53.27213203, 44.72588407, 45.27213056, 37.49340653, 43.3199716 , 0. , 15.25506793, 27.16951934, 9.39975891, 23.29823883, 27.4569762 , 36.66493295, 25.41404487, 62.30658435, 59.49162941, 54.97861729, 62.27438946], [ 56.12197041, 54.23178552, 44.09602552, 54.43942811, 39.94535473, 39.11436642, 35.05276887, 39.04821937, 46.66219584, 160.97214311, 43.92941224, 54.78849677, 43.28927735, 43.58825567, 33.01661549, 41.56030427, 15.25506793, 0. , 20.35750684, 11.75277613, 28.77849964, 20.38170604, 28.03229244, 23.52809911, 65.79526165, 61.03327922, 53.18200867, 64.68851869], [ 46.6306051 , 44.83854602, 34.43324484, 44.77225936, 32.40097031, 31.50167232, 25.9536145 , 31.42659777, 49.31545741, 162.09566192, 42.36351092, 56.47060418, 35.62533925, 35.64040039, 23.88261331, 33.26418178, 27.16951934, 20.35750684, 0. , 28.47386998, 31.05543352, 24.14340374, 13.61655662, 27.08986991, 67.12165227, 62.16957078, 50.29552178, 65.02769493], [ 59.1941265 , 57.43040333, 50.05861466, 57.62475899, 42.02489702, 40.98442719, 41.17658427, 40.92228771, 41.64553807, 157.8989561 , 44.30355156, 52.11976076, 44.87291313, 45.36910866, 39.18968134, 43.64184861, 9.39975891, 11.75277613, 28.47386998, 0. , 26.44661684, 26.10484364, 36.53419784, 22.68468582, 61.59584252, 58.03516782, 53.81975332, 61.27832302], [ 44.66173968, 43.66256723, 38.01670253, 43.07753295, 36.45887744, 35.81116651, 34.44990808, 35.4526166 , 47.70710761, 159.96856599, 47.17147993, 54.53343724, 37.23850825, 36.52130938, 30.91574375, 34.76399127, 23.29823883, 28.77849964, 31.05543352, 26.44661684, 0. , 12.58807854, 22.92923394, 6.76202654, 60.62057734, 58.4789755 , 55.54830708, 62.65780807], [ 42.89555071, 41.07342553, 33.55169178, 41.07062472, 33.04496766, 32.41245948, 29.47454279, 32.15452408, 48.88045288, 159.47298318, 44.42038965, 54.34704183, 34.67224658, 33.92944143, 25.99260308, 31.73095692, 27.4569762 , 20.38170604, 24.14340374, 26.10484364, 12.58807854, 0. , 16.68014741, 7.36545979, 61.86498436, 57.14215663, 52.81936503, 62.53688323], [ 41.77752184, 39.88392766, 26.86238671, 39.71176769, 30.7060353 , 30.35891937, 24.87492583, 29.82652693, 54.40050937, 164.47195627, 47.77566898, 60.8553995 , 33.20937602, 32.66279228, 22.01089353, 29.88389113, 36.66493295, 28.03229244, 13.61655662, 36.53419784, 22.92923394, 16.68014741, 0. , 19.19139784, 69.33446311, 64.09019718, 53.25118704, 68.80945012], [ 43.11120063, 41.84901395, 36.62259533, 41.51438582, 34.8030323 , 34.20251382, 32.82868174, 33.86555348, 48.27914604, 160.01193211, 45.78773909, 53.84452399, 35.76602818, 35.06526933, 29.35476974, 33.24355623, 25.41404487, 23.52809911, 27.08986991, 22.68468582, 6.76202654, 7.36545979, 19.19139784, 0. , 61.08620107, 57.25864273, 54.34578176, 62.16146566], [ 68.32061644, 66.85474201, 71.81566052, 68.49775088, 56.05818707, 55.52899446, 59.69310905, 54.77179363, 38.55325502, 166.1647325 , 46.84575567, 44.92297076, 51.58668133, 54.7645009 , 56.48611768, 54.09424466, 62.30658435, 65.79526165, 67.12165227, 61.59584252, 60.62057734, 61.86498436, 69.33446311, 61.08620107, 0. , 27.83323415, 43.3144985 , 32.03301218], [ 65.14603125, 63.82797017, 67.19749864, 65.55193363, 51.97626164, 51.38360849, 54.87867452, 50.687827 , 45.88218146, 160.31935393, 50.56998455, 52.30116582, 45.78590302, 46.97234601, 51.2714387 , 47.62494125, 59.49162941, 61.03327922, 62.16957078, 58.03516782, 58.4789755 , 57.14215663, 64.09019718, 57.25864273, 27.83323415, 0. , 36.58015997, 32.9207959 ], [ 65.68641834, 63.77509416, 53.84269777, 63.789327 , 50.11901623, 49.70551132, 44.27126243, 49.22438041, 51.01618252, 166.61501547, 33.47990847, 51.65090646, 53.26689779, 53.54782816, 42.21734349, 51.56686678, 54.97861729, 53.18200867, 50.29552178, 53.81975332, 55.54830708, 52.81936503, 53.25118704, 54.34578176, 43.3144985 , 36.58015997, 0. , 38.66079144], [ 78.55732438, 77.19133046, 72.20182482, 78.25638756, 61.22402847, 60.93167753, 60.6742164 , 60.5146899 , 47.32256434, 175.77625682, 41.40076211, 36.93469696, 62.22876967, 64.20936251, 58.93869991, 63.62161223, 62.27438946, 64.68851869, 65.02769493, 61.27832302, 62.65780807, 62.53688323, 68.80945012, 62.16146566, 32.03301218, 32.9207959 , 38.66079144, 0. ]])
ind = np.array([i for i in range(letter_dist.shape[0]) if i != 9])
letter_dist = letter_dist[np.ix_(ind,ind)]

res = {}
for sym in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
  for f in fonts:
    res[(f, sym)] = sym

Z = cmds(letter_dist)
plt.scatter(*Z.T)
for i,(k,v) in enumerate(res.items()):
  if i in ind: 
    print(v)
    idx = np.flatnonzero(ind == i).item()
    plt.text(Z[idx,0], Z[idx,1], str(v))


