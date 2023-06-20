import numpy as np
from pbsig.linalg import * 
from pbsig.persistence import sliding_window, sw_parameters
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
output_notebook(verbose=False)
# from pbsig.vis import 

sw_f = lambda t: np.cos(t) + np.cos(3*t)
dom = np.linspace(0, 12*np.pi, 1200)

p = figure(width=400, height=200)
p.line(dom, sw_f(dom))
show(p)

# st = SimplexTree(complete_graph(X.shape[0]))
# st.expand(2)
# S = st

N, M = 20, 24
SW = sliding_window(sw_f, bounds=(0, 12*np.pi))
d, tau = sw_parameters(bounds=(0,12*np.pi), d=M, L=6)
#S = delaunay_complex(F(n=N, d=M, tau=tau))
X = SW(n=N, d=M, tau=tau)
# r = enclosing_radius(X)*0.60
# S = rips_complex(X, r, 2)
show(plot_complex(S, X[:,:2]))

## Plot 
scatters = []
for t in np.linspace(0.50*tau, 1.50*tau, 10):
  X_delay = SW(n=N, d=M, tau=t)
  p = figure(width=150, height=150, toolbar_location=None)
  p.scatter(*pca(X_delay).T)
  scatters.append(plot_complex(S, pos=pca(X_delay), width=125, height=125))
show(row(*scatters))

from pbsig.persistence import ph
from pbsig.vis import plot_dgm
K = filtration(S, f=flag_weight(X))
dgm = ph(K, engine="dionysus")
plot_dgm(dgm[1])

from pbsig.betti import MuSignature, mu_query
from pbsig.linalg import * 
R = np.array([4, 4.5, 6.5, 7.5])
T_dom = np.append(np.linspace(0.87*tau, tau, 150, endpoint=False), np.linspace(tau, tau*1.12, 150, endpoint=False))
t_family = [flag_weight(SW(n=N, d=M, tau=t)) for t in T_dom]

MU_f = mu_query(S, R=R, f=flag_weight(SW(n=N, d=M, tau=tau)), p=1, form="array")
MU_f(smoothing=None)
MU_f(smoothing=False)
MU_f(smoothing=True)
MU_f(smoothing=huber())
MU_f(smoothing=soft_threshold())
MU_f(smoothing=moreau(), terms=True)


sig = MuSignature(S, family=t_family, R=R, p=1, form="array")
sig.precompute(w=0.0, normed=False, progress=True)

max(sig(smoothing=False))


## Cone the complex
from scipy.spatial.distance import pdist 
from pbsig.betti import cone_weight
X = SW(n=N, d=M, tau=tau)
diam = max(pdist(X))
r = enclosing_radius(X)*0.60
S = rips_complex(X, r, 2)
S = SetComplex(S)
sv = X.shape[0] # special vertex
S.add([sv])
S.update([s + [sv] for s in faces(S, 1)])
K = filtration(S, f=cone_weight(X, sv, 0.0, diam/2))

## Measure the persistence
from pbsig.persistence import ph
from pbsig.vis import plot_dgm
dgm = ph(K, engine="cpp")
plot_dgm(dgm[1])

# ## Verify mu queries 
# from pbsig.betti import mu_query, mu_query_mat
# R = np.array([4, 4.2, 4.8, 5.2])
# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array'))
# R = np.array([3.8, 4.0, 4.8, 5.2])
# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array'))
# R = np.array([4.2, 4.4, 4.8, 5.2])
# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array'))
# R = np.array([4, 4.2, 5.1, 5.2])
# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array'))
# R = np.array([4, 4.2, 4.8, 4.85])
# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array'))
# R = np.array([4, 4.2, 4.8, 5.2])
# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array'))


T_dom = np.append(np.linspace(0.87*tau, tau, 150, endpoint=False), np.linspace(tau, tau*1.12, 150, endpoint=False))
t_family = [cone_weight(SW(n=N, d=M, tau=t), sv, 0.0,diam/2) for t in T_dom]

## Plot cone weights over time 
p = figure(width=400, height=250)
for s in faces(K):
  s_vals = np.array([f(s) for f in t_family])
  p.line(T_dom, s_vals)
# p.legend.location = "bottom_right"
show(p) 

## Precompute the multiplicities for R
from pbsig.betti import MuSignature
from pbsig.linalg import * 
R = np.array([4, 4.2, 4.8, 5.2])
sig = MuSignature(S, family=t_family, R=R, p=1, form="array")
sig.precompute(w=0.0, normed=False)

## Unnormalized rank relaxations 
p = figure(width=400, height=250)
p.line(T_dom, sig(smoothing=False), legend_label="Nuclear", line_color="orange")
p.line(T_dom, sig(smoothing=huber(delta=0.10)), legend_label="Huber-0.10", line_color="purple")
p.line(T_dom, sig(smoothing=sgn_approx(eps=0.05, method=0, p=1.5)), legend_label="Sgn_Approx", line_color="blue")
p.line(T_dom, sig(smoothing=soft_threshold(t=0.01)), legend_label="Soft", line_color="red")
p.line(T_dom, sig(smoothing=None), legend_label="Rank", line_color="black")
# p.line(T_dom, sig(smoothing=True), legend_label="Huber", line_color="purple")
show(p)

from pbsig.linalg import * 
sig_norm = MuSignature(S, family=t_family, R=R, p=1, form="array")
sig_norm.precompute(w=0.0, normed=True)

p = figure(width=400, height=250)
p.line(T_dom, sig_norm(smoothing=False), legend_label="Nuclear", line_color="orange")
p.line(T_dom, sig_norm(smoothing=huber(delta=0.10)), legend_label="Huber-0.10", line_color="purple")
p.line(T_dom, sig_norm(smoothing=sgn_approx(eps=0.05, method=0, p=1.5)), legend_label="Sgn_Approx", line_color="blue")
p.line(T_dom, sig_norm(smoothing=soft_threshold(t=1.55)), legend_label="Soft", line_color="red")
p.line(T_dom, sig_norm(smoothing=moreau(t=.05)), legend_label="Moreau", line_color="green")
p.line(T_dom, sig_norm(smoothing=None), legend_label="Rank", line_color="black")
# p.line(T_dom, sig(smoothing=True), legend_label="Huber", line_color="purple")
show(p)


from pbsig.linalg import * 
sig_norm_exp = MuSignature(S, family=t_family, R=R, p=1, form="array")
sig_norm_exp.precompute(w=0.10, normed=True)

p = figure(width=400, height=250)
p.line(T_dom, sig_norm_exp(smoothing=False), legend_label="Nuclear", line_color="orange")
p.line(T_dom, sig_norm_exp(smoothing=huber(delta=0.10)), legend_label="Huber-0.10", line_color="purple")
p.line(T_dom, sig_norm_exp(smoothing=sgn_approx(eps=0.05, method=0, p=1.5)), legend_label="Sgn_Approx", line_color="blue")
p.line(T_dom, sig_norm_exp(smoothing=soft_threshold(t=3.5)), legend_label="Soft", line_color="red")
#p.line(T_dom, sig_norm_exp(smoothing=moreau(t=3.5)), legend_label="Moreau", line_color="green")
p.line(T_dom, sig_norm_exp(smoothing=None), legend_label="Rank", line_color="black")
# p.line(T_dom, sig(smoothing=True), legend_label="Huber", line_color="purple")
show(p)

## Ensure the rank is correct between all three 
assert np.allclose(sig_norm(smoothing=None), sig_norm_exp(smoothing=None))
assert np.allclose(sig(smoothing=None), sig_norm_exp(smoothing=None))


from pbsig.linalg import * 
sig_norm_exp = MuSignature(S, family=t_family, R=R, p=1, form="array")
sig_norm_exp.precompute(w=0.50, normed=True)

p = figure(width=400, height=250)
p.line(T_dom, sig_norm_exp(smoothing=False), legend_label="Nuclear", line_color="orange")
p.line(T_dom, sig_norm_exp(smoothing=huber(delta=0.50)), legend_label="Huber-0.10", line_color="purple")
p.line(T_dom, sig_norm_exp(smoothing=sgn_approx(eps=0.05, method=0, p=1.5)), legend_label="Sgn_Approx", line_color="blue")
p.line(T_dom, sig_norm_exp(smoothing=soft_threshold(t=0.85)), legend_label="Soft", line_color="red")
p.line(T_dom, sig_norm_exp(smoothing=moreau(t=0.85)), legend_label="Moreau", line_color="green")
p.line(T_dom, sig_norm_exp(smoothing=None), legend_label="Rank", line_color="black")
# p.line(T_dom, sig(smoothing=True), legend_label="Huber", line_color="purple")
show(p)

## Make animation 

X = np.random.uniform(size=(5,5))
M = X @ X.T 
ew, U = np.linalg.eigh(M)
assert np.allclose(M - (U @ diags(ew) @ U.T), 0.0)
A = U @ diags(soft_threshold(ew, 0.50)) @ U.T
m_env = sum(abs(ew)) + (1/(2*0.50))*np.linalg.norm(A - M, 'fro')**2

sum(ew) + sum(np.where(ew >= 0.50, 0.5, ew)**2) ## here it is 


from pbsig.linalg import * 
prox_nuclear(X @ X.T)




## Debug w
from pbsig.utility import *
p = 1
w = 0.30
ti = np.argmax(abs(sig_norm_exp(smoothing=True)))
t, cone_f = T_dom[ti], t_family[ti]
(i,j,k,l) = R
pw = np.array([cone_f(s) for s in faces(S, p)])
qw = np.array([cone_f(s) for s in faces(S, p+1)])
delta = np.finfo(float).eps                       
fi = smooth_upstep(lb = i, ub = i+w)(pw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
fj = smooth_upstep(lb = j, ub = j+w)(pw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
fk = smooth_dnstep(lb = k-w, ub = k+delta)(qw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
fl = smooth_dnstep(lb = l-w, ub = l+delta)(qw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]

## Ensure the signs are all equivalent
assert all(np.sign(smooth_upstep(lb = i, ub = i+0.0)(pw)) == np.sign(fi))
assert all(np.sign(smooth_upstep(lb = j, ub = j+0.0)(pw)) == np.sign(fj))
assert all(np.sign(smooth_dnstep(lb = k-0.0, ub = k+delta)(qw)) == np.sign(fk))
assert all(np.sign(smooth_dnstep(lb = l-0.0, ub = l+delta)(qw)) == np.sign(fl))

## Compute the multiplicities 
pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
EW = [None]*4
D = boundary_matrix(S, p=p+1)
for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
  di = (D @ diags(J) @ D.T).diagonal()
  I_norm = pseudo(np.sqrt(di))
  L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
  EW[cc] = smooth_rank(L, pp=1.0, smoothing=None) # sum(np.linalg.eigvalsh(L.todense())) 

## Try fixing the degree
fj = smooth_upstep(lb = j, ub = j+w)         
fl = smooth_dnstep(lb = l-w, ub = l+delta)
I,J = fj(pw), fl(qw)
edges = list(faces(S,p))
edge_weights = np.zeros(len(edges))
for coface in faces(S, p+1):
  e_ind = np.array([edges.index(e) for e in coface.boundary()])
  edge_weights[e_ind] += fl(cone_f(coface)).item()
assert np.allclose((D @ diags(J) @ D.T).diagonal() - edge_weights, 0)

di = (D @ diags(J) @ D.T).diagonal()
#np.linalg.eigvalsh((diags(1.0/di) @ D @ diags(J) @ D.T).todense())
L = diags(pseudo(np.sqrt(di))) @ D @ diags(J) @ D.T @ diags(pseudo(np.sqrt(di)))
np.linalg.eigvalsh(L.todense())






C = [list(S.cofaces(e)) for e in faces(S, p)]
deg_F = np.array([sum([cone_f(c) for c in CF if dim(c) == p+1]) for CF in C])
deg_F

# np.linalg.eigvalsh((D @ diags(J) @ D.T).todense())

# print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=t), sv, 0.0,diam/2), p=1, form='array'))

## Ensure we're doing the same thing
# mu_truth = [mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=t), sv, 0.0,diam/2), p=1, form='array') for t in T_dom]
# assert np.allclose(np.array(mu_truth) - sig(smoothing=None), 0.0)


# print(mu_query(K, R=R, f=t_family[0], p=1, form='array', terms=True))

# from scipy.sparse import lil_array
# T1 = lil_array((len(t_family), card(S, 1)), dtype=np.float64)
# for cc in range(T1.shape[0]):
#   v = sig._terms[0][cc]
#   T1[cc,:len(v)] = v
# T1 = T1.tocsr()

# sum(sig._terms[2][0] > 1e-12)


## Verify spectrum 
# R = np.array([4, 4.2, 4.8, 5.2])
# LM  = mu_query_mat(K, f=cone_weight(SW(n=N, d=M, tau=t), sv, 0.0, diam/2), R=R, p=1, form='array')

for L in LM:
  #print(max(np.linalg.eigvalsh(L.todense())))
  #print(max(np.linalg.eigvalsh(L.todense())))
  print(np.linalg.matrix_rank(L.todense()))

## Verify queries match up with matrices
print(mu_query(K, R=R, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), p=1, form='array', terms=True))

#%%  Test the multiplicity queries with the coned complex
# from pbsig.betti import mu_query
# R = np.array([-np.inf, 5, 15, np.inf])
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=False, terms=True, form='lo'))
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=False, terms=True, form='array'))



# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='lo'))
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='array'))

# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='lo', raw=True)[2])
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='array', raw=True)[2])

# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=False, terms=True, form='lo'))
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=False, terms=True, form='array'))

# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=True, terms=True, form='lo'))
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=True, terms=True, form='array'))

# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=True, terms=True, form='lo'))
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=True, terms=True, form='array'))

# import bokeh 
# from bokeh.plotting import figure, show 
# rr = [sum(wut/(wut + t)) for t in np.linspace(1, 1e-50, 1000, endpoint=True)]
# p = figure(width=300, height=300)
# p.line(np.arange(len(rr)), rr)
# show(p)


# R = np.array([-np.inf, 4, 15, np.inf])
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=True))
# R = np.array([-np.inf, 5, 15, np.inf])


# eigh2rank(mat2eigh(mat2dense(time2mat(t))))
# # mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(0.000000001, 1.0, 0), terms=True)
# mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing = None, terms = True, sqrt=False)
# MM = mat2eigh(mat2dense(time2mat(tau)))
# sum(np.sqrt(np.maximum(MM[2], 0.0)))

# mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing = None, terms = True, sqrt=False, raw=True)[2]
# MM[2]
# sum(np.sqrt(np.maximum(MM[2], 0.0)))

# R = np.array([-np.inf, 4, 15, np.inf])
# LM = mu_query_mat(K, R=R, f=cone_weight(X,sv), p=1, form='array')
#time2mat = lambda t: mu_query(K, f=cone_weight(SW(n=N, d=M, tau=t), sv), R=R, p=1, smoothing=(1e-4, 1.0, 0), sqrt=True, form='lo')

## MAD https://arxiv.org/pdf/2202.11014.pdf
# print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-4, 1.0, 0), sqrt=True, terms=False, form='array'))
# from pbsig.betti import *
# assert np.allclose(X - SW(n=N, d=M, tau=tau), 0.0)
# R = np.array([4, 4.2, 4.8, 5.2])
# print(mu_query(K, R=R, f=cone_weight(X,sv, 0.0, diam/2), p=1, smooth=False, sqrt=True, terms=True, form='array'))
# DL = mu_query_mat(K, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0, diam/2), R=R, p=1, form='array')
# # np.linalg.matrix_rank(DL[0].todense()) - np.linalg.matrix_rank(DL[1].todense()) - np.linalg.matrix_rank(DL[2].todense()) + np.linalg.matrix_rank(DL[3].todense())

## 
from pbsig.linalg import prox_nuclear


R = np.array([4, 4.2, 4.8, 5.0]) # TODO: <--- add w term here 
time2mat = lambda t: mu_query_mat(K, f=cone_weight(SW(n=N, d=M, tau=t), sv, 0.0, diam/2), R=R, p=1, form='array')
mat2dense = lambda L: [l.todense() for l in L]
mat2eigh = lambda L: [np.linalg.eigvalsh(l) for l in L]
mat2rank = lambda L: [np.linalg.matrix_rank(l) for l in L]
mat2mora = lambda L, t: [prox_nuclear(l, t)[1] for l in L]
eigh2nucl = lambda E: [sum(np.sqrt(np.maximum(0.0, e))) for e in E]
eigh2rank = lambda E: [sum(~np.isclose(e, 0.0)) for e in E]
terms2mu = lambda T: sum([s*term for term, s in zip(T, [1,-1,-1,1])])
rank_obj = lambda t: terms2mu(mat2rank(mat2dense(time2mat(t))))
nucl_obj = lambda t: terms2mu(eigh2nucl(mat2eigh(mat2dense(time2mat(t)))))
# prox_obj = lambda t: terms2mu(eigh2nucl(mat2eigh(mat2dense(time2mat(t)))))
mora_obj = lambda t,s: terms2mu(mat2mora(mat2dense(time2mat(t)), s))
# moreau_obj = lambda t: terms2mu(eigh2nucl(mat2eigh(mat2dense(time2mat(t)))))

## Compute the various objectives
T_dom = np.append(np.linspace(0.87*tau, tau, 150, endpoint=False), np.linspace(tau, tau*1.12, 150, endpoint=False))
nuclear_query = np.array([nucl_obj(t) for t in T_dom])
rank_query = np.array([rank_obj(t) for t in T_dom])
moreau1_query = np.array([mora_obj(t, 0.01) for t in T_dom])
moreau2_query = np.array([mora_obj(t, 0.10) for t in T_dom])
moreau3_query = np.array([mora_obj(t, 1.00) for t in T_dom])

## show the objective 
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.models import Span 
p = figure(width=400, height=200)
p.line(T_dom, nuclear_query, color='blue', line_width=2.15)
p.scatter(T_dom, nuclear_query, color='purple', size=3.15)
p.scatter(T_dom, rank_query, color='red', size=1.75)
p.scatter(T_dom, moreau1_query, color='orange', size=3.15)
p.scatter(T_dom, moreau2_query, color='pink', size=3.15)
p.scatter(T_dom, moreau3_query, color='green', size=3.15)
vline = Span(location=tau, dimension='height', line_color='red', line_width=1)
p.add_layout(vline)
show(p)



# from pbsig.linalg import prox
# from findiff import FinDiff, coefficients, Coefficient
# T0 = time2mat(tau)
# T1 = time2mat(0.90*tau)
# T1[0].todense() - T0[0].todense()
# dt = tau-0.90*tau
# df_dt = FinDiff(0, tau-0.90*tau, 1, acc=4)
# df_dt(OBJ)



# from pbsig.betti import MuSignature
# T_dom = np.linspace(0.50*tau, 1.50*tau, 100)
# family = [cone_weight(SW(n=N, d=M, tau=t)) for t in T_dom]
# sig = MuSignature(K, family=family, R=R, p=1)
# sig.precompute()
# q = sig(tau, smoothing=(1e-14, 1.0, 0))


prox(LM[0])

alpha, tau0, t1, T = 0.1, 0.50*tau, 1.0, 1.50*tau

ew, ev = np.linalg.eigh(LM[0].todense())



# dgms = [ph(filtration(S, f=cone_weight(SW(n=N, d=M, tau=t))))[1] for t in T_dom]
# for dgm in dgms:
#   valid_birth = np.logical_and(dgm['birth'] >= R[0], dgm['birth'] <= R[1])
#   valid_death = np.logical_and(dgm['death'] >= R[2], dgm['death'] <= R[3])
#   print(sum(np.logical_and(valid_birth, valid_death)))

## Choose a box, show its rank over vineyards 
# LM = mu_query_mat(K, f=cone_weight(X, sv), R=R, p=1, terms=False)

# # eigh2rank(mat2eigh(mat2dense(L)))
# terms2mu(eigh2nucl(mat2eigh(mat2dense(LM))))
# terms2mu(eigh2rank(mat2eigh(mat2dense(LM))))
#L = mu_query_mat(K, f=cone_weight(SW(n=N, d=M, tau=tau), sv, 0.0,diam/2), R=R, p=1, form='array')

# 0-2-28+31 == 1
# from pbsig.betti import mu_query
# print(mu_query(K, R=R, f=cone_weight(X,sv,0.0,diam/2), p=1, smoothing=None, sqrt=False, terms=True, form='lo'))
# print(mu_query(K, R=R, f=cone_weight(X,sv,0.0,diam/2), p=1, smoothing=None, sqrt=False, terms=True, form='array'))

# w = mu_query(K, R=R, f=cone_weight(X,sv,0.0,diam/2), p=1, smoothing=None, sqrt=False, terms=True, raw=True, form='array')
# C = 1.0/np.exp(np.linspace(1,32, 100))

# nr = [sum(w[3]/(w[3] + c)) for c in C]
# values, counts = np.unique(np.round(nr).astype(int), return_counts=True)
# values[np.argmax(counts)]
# p = figure(width=400, height=200)
# p.line(np.arange(len(nr)), nr)
# show(p)

# kneedle = KneeLocator(np.arange(len(nr)), nr, S=100, curve="concave", direction="increasing")

# vline = Span(location=kneedle.knee, dimension='height', line_color='red', line_width=1)
# p.add_layout(vline)
# show(p)
# assert np.allclose(sig.elementwise_row_sum(sig._Terms[0], lambda x: x) - sig._Terms[0].sum(axis=1), 0)
# T = sig._Terms[0].copy()
# h = huber(delta=0.25)
# T.data = h(T.data+1)
# assert np.allclose(sig.elementwise_row_sum(T, lambda x: x) - sig.elementwise_row_sum(sig._Terms[0], lambda x: h(x+1)), 0)


# T = sig._Terms[0].copy()
# h = soft_threshold(t=0.05)
# T.data = h(T.data+1)
# assert np.allclose(sig.elementwise_row_sum(T, lambda x: x) - sig.elementwise_row_sum(sig._Terms[0], lambda x: h(x+1)), 0)

# T = sig._Terms[0].copy()
# h = sgn_approx(eps=1e-5, p=1.2)
# T.data = h(T.data+1)
# assert np.allclose(sig.elementwise_row_sum(T, lambda x: x) - sig.elementwise_row_sum(sig._Terms[0], lambda x: h(x+1)), 0)
