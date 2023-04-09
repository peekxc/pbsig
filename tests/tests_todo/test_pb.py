import numpy as np 
from itertools import product
from pbsig import rotate_S1
from pbsig.simplicial import delaunay_complex
from pbsig.betti import lower_star_betti_sig, persistent_betti
from pbsig.utility import is_sorted

X = np.random.uniform(size=(38,2))
K = delaunay_complex(X)
F = list(rotate_S1(X, 32, include_direction=False))
E = K['edges']

## Setup lex boundary matrix
fv, ev = F[0], F[0][E].max(axis=1)
V_names = [str(v) for v in K['vertices']]
E_names = [str(tuple(e)) for e in K['edges']]
VF0 = dict(zip(V_names, fv))
EF0 = dict(zip(E_names, ev))
D0, D1 = boundary_matrix(K, p=(0,1))

# D1 = D1[np.ix_(np.argsort(fv), np.argsort(ev))]
# np.linalg.matrix_rank(D1_f.A)
# np.linalg.matrix_rank(D1.A)
# np.set_printoptions(edgeitems=30, linewidth=100000)
# D1_f[ii:, :jj]
vf = F[fi]
ef = vf[E].max(axis=1)

vf_s, ef_s = np.array(sorted(vf)), np.array(sorted(ef))
D1_f = D1[np.ix_(np.argsort(vf), np.argsort(ef))].copy()
for i,j in product(vf, ev):
  if i < j:
    ## This requires D1 to be in sorted filtration order
    assert is_sorted(vf_s) and is_sorted(ef_s)
    ii, jj = sum(vf_s <= i), sum(ef_s <= j) # 1-based, inclusive
    p1 = np.array(persistent_betti(D0, D1_f, i=ii, j=jj, summands=True))
    #D1_f.todense().astype(int) % 2

    ## This does not 
    nv = len(fv)
    #D1_t = lower_star_betti_sig([fv], p_simplices=E, nv=nv, a=i, b=j, method="rank", keep_terms=True)
    #BB = D1_t[np.ix_(np.argsort(fv), np.argsort(ev))].A.astype(int) % 2
    p2 = lower_star_betti_sig([fv], p_simplices=E, nv=nv, a=i, b=j, method="rank", keep_terms=True).flatten().astype(int)
    if p1[0] == 0 and p2[0] == 0:
      continue
    assert all(abs(p1) == abs(p2)), "Failed assertion"

## Check from every angle
D0, D1 = boundary_matrix(K, p=(0,1))
E = K['edges']
PB_normal = {}
for fi, vf in enumerate(F):
  nv = D0.shape[1]
  ef = vf[E].max(axis=1)
  vf_s, ef_s = np.array(sorted(vf)), np.array(sorted(ef))
  D1_f = D1[np.ix_(np.argsort(vf), np.argsort(ef))].copy()
  for i,j in product(vf, ef):
    if i < j:
      assert is_sorted(vf_s) and is_sorted(ef_s)
      ii, jj = sum(vf_s <= i), sum(ef_s <= j) # 1-based, inclusive
      if fi == 1 and ii == 3 and jj == 12:
        raise ValueError("a")
      PB_normal[(fi,i,j)] = (np.array(persistent_betti(D0, D1_f, i=ii, j=jj, summands=True)))

## Check from every angle: relaxation
PB_sigs = {}
for fi, vf in enumerate(F):
  nv = D0.shape[1]
  ef = vf[E].max(axis=1)
  for i,j in product(vf, ef):
    if i < j:
      sig = abs(lower_star_betti_sig([vf], E, nv=nv, a=i, b=j, method="rank", keep_terms=True).flatten().astype(int))
      PB_sigs[(fi,i,j)] = sig
      
PB_normal_terms = np.array(list(PB_normal.values()))
PB_sig_terms = np.array(list(PB_sigs.values()))
error = (abs(PB_normal_terms) - abs(PB_sig_terms)).sum(axis=1)
assert(all(error == 0))

assert list(PB_sigs.keys()) == list(PB_normal.keys())
for fi, i, j in PB_sigs.keys():
  assert all(abs(PB_sigs[(fi,i,j)]) == abs(PB_normal[(fi,i,j)]))
vf = F[fi]

fi,i,j = list(PB_sigs.keys())[0]
# [(PB_sigs[k], PB_normal[k]) for k in PB_sigs.keys()]

wrong_idx = min(np.flatnonzero(error != 0))
fi, i, j = list(PB_sigs.keys())[wrong_idx]

PB_normal_terms[wrong_idx,:]
PB_sig_terms[wrong_idx,:]


np.linalg.matrix_rank(D1_f[ii:,:][:,:jj].tocsc().A)
D1_f[ii:,:jj]
sum(abs(np.array([s*af*bf if af*bf > 0 else 0.0 for (s, af, bf) in zip(D1_nz_pattern, A_exc, B_inc)])))

for i in range(10000):
  rp = np.random.permutation(BB.shape[0]).astype(int)
  cp = np.random.permutation(BB.shape[1]).astype(int)
  assert np.linalg.matrix_rank(BB[np.ix_(rp,cp)]) == 2

## Individual tests
sum(fv <= i)
