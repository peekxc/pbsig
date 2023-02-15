import numpy as np
from pbsig.linalg import * 
from pbsig.persistence import sliding_window, sw_parameters
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
output_notebook(verbose=False)
from pbsig.vis import plot_complex

f = lambda t: np.cos(t) + np.cos(3*t)
dom = np.linspace(0, 12*np.pi, 1200)

p = figure(width=400, height=200)
p.line(dom, f(dom))
show(p)


N, M = 20, 24
SW = sliding_window(f, bounds=(0, 12*np.pi))
d, tau = sw_parameters(bounds=(0,12*np.pi), d=M, L=6)
#S = delaunay_complex(F(n=N, d=M, tau=tau))
X = SW(n=N, d=M, tau=tau)
r = enclosing_radius(X)*0.60
S = rips_complex(X, r, 2)
show(plot_complex(S, X[:,:2]))

## Plot 
scatters = []
for t in np.linspace(0.50*tau, 1.50*tau, 10):
  X_delay = SW(n=N, d=M, tau=t)
  p = figure(width=150, height=150, toolbar_location=None)
  p.scatter(*pca(X_delay).T)
  scatters.append(plot_complex(S, pos=pca(X_delay), width=125, height=125))
show(row(*scatters))


## Cone the complex
from pbsig.betti import cone_weight
X = SW(n=N, d=M, tau=tau)
S = rips_complex(X, r, 2)
S = SetComplex(S)
sv = X.shape[0] # special vertex
S.add([sv])
S.update([s + [sv] for s in faces(S, 1)])
K = filtration(S, f=cone_weight(X, sv))

## Measure the persistence
from pbsig.persistence import ph
from pbsig.vis import plot_dgm
dgm = ph(K)
plot_dgm(dgm[1])

## Test the multiplicity queries with the coned complex
from pbsig.betti import mu_query
R = np.array([-np.inf, 5, 15, np.inf])
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=False, terms=True, form='lo'))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=False, terms=True, form='array'))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='lo'))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='array'))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=False, terms=True, form='lo', raw=True)[2])
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=True, terms=True, form='lo', raw=True))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=None, sqrt=True, terms=True, form='array'))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=True, terms=True, form='lo'))
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=True, terms=True, form='array'))

R = np.array([-np.inf, 4, 15, np.inf])
print(mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(1e-10, 1.0, 0), sqrt=True))
R = np.array([-np.inf, 5, 15, np.inf])

## WHYYYYY 
eigh2rank(mat2eigh(mat2dense(time2mat(t))))
# mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing=(0.000000001, 1.0, 0), terms=True)
mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing = None, terms = True, sqrt=False)
MM = mat2eigh(mat2dense(time2mat(tau)))
sum(np.sqrt(np.maximum(MM[2], 0.0)))

mu_query(K, R=R, f=cone_weight(X,sv), p=1, smoothing = None, terms = True, sqrt=False, raw=True)[2]
MM[2]
# sum(np.sqrt(np.maximum(MM[2], 0.0)))


## MAD https://arxiv.org/pdf/2202.11014.pdf
from pbsig.betti import *
time2mat = lambda t: mu_query_mat(K, f=cone_weight(SW(n=N, d=M, tau=t), sv), R=R, p=1)
mat2dense = lambda L: [l.todense() for l in L]
mat2eigh = lambda L: [np.linalg.eigvalsh(l) for l in L]
eigh2nucl = lambda E: [sum(np.sqrt(np.maximum(0.0, e))) for e in E]
eigh2rank = lambda E: [sum(~np.isclose(e, 0.0)) for e in E]
terms2mu = lambda T: sum([s*term for term, s in zip(T, [1,-1,-1,1])])
rank_obj = lambda t: terms2mu(eigh2rank(mat2eigh(mat2dense(time2mat(t)))))
nucl_obj = lambda t: terms2mu(eigh2nucl(mat2eigh(mat2dense(time2mat(t)))))

## show the objective 
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.models import Span 
T_dom = np.linspace(0.50*tau, 1.50*tau, 100)
p = figure(width=400, height=200)
p.scatter(T_dom, [nucl_obj(t) for t in T_dom], color='blue')
p.scatter(T_dom, [rank_obj(t) for t in T_dom], color='red')
vline = Span(location=tau, dimension='height', line_color='red', line_width=1)
p.add_layout(vline)
show(p)

from pbsig.linalg import prox
from findiff import FinDiff, coefficients, Coefficient
T0 = time2mat(tau)
T1 = time2mat(0.90*tau)
T1[0].todense() - T0[0].todense()
dt = tau-0.90*tau
df_dt = FinDiff(0, tau-0.90*tau, 1, acc=4)
df_dt(OBJ)



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
