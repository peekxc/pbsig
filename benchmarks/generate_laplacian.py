import numpy as np 
from splex import *
from more_itertools import collapse 

# %% 
from pbsig.perfect_hash import perfect_hash, perfect_hash_dag
# perfect_hash(er, output = "expression", solver = "linprog", k_min = 3, k_max = 15, n_tries = 100, n_prime = 1000)
  
g, expr = perfect_hash_dag(er, output = "expression", mult_max = 2.5, n_tries = 1500, n_prime = 1000)
# all([eval(expr, None, dict(x=r))==i for i,r in enumerate(er)])
# sum(g != 0)

# np.savetxt(X=g, fname="/Users/mpiekenbrock/pbsig/data/er_g_colex_500.txt", fmt="%d")
# (g[((5618*x+9597)%11699)%6139]+g[((3170*x+7593)%8627)%6139])%6139


np.random.seed(1234)
g, expr, G = perfect_hash_dag([0, 5, 13, 45, 82], mult_max = 1.5, output="expression")
print(G.adj)
print(expr)
# g = array([0, 3, 3, 4, 2, 0])
# wrong: g[ 0, 1, 3, 4, 2, 0, ]
# (g[((466*x+462)%523)%6]+g[((133*x+215)%281)%6])%6

# NG = 5
# array([0, 2, 0, 1, 0])
# (g[((362*x+409)%449)%5]+g[((307*x+110)%359)%5])%5

f1 = lambda x: ((362*x+409)%449)%5
f2 = lambda x: ((307*x+110)%359)%5


# %% Large data set (colex)
np.random.seed(1234)
X = np.random.uniform(size=(250,2))
R = rips_complex(X, radius=0.10)
R.expand(2)
er = rank_combs(faces(R, 1), n=X.shape[0], order="colex")
tr = rank_combs(faces(R, 2), n=X.shape[0], order="colex")
# np.savetxt(X=er, fname="/Users/mpiekenbrock/pbsig/data/edge_ranks_colex_500.txt", fmt="%d")
# np.savetxt(X=tr, fname="/Users/mpiekenbrock/pbsig/data/triangle_ranks_colex_500.txt", fmt="%d")
s_labels = np.fromiter(collapse(list(faces(R, 2))), dtype=np.uint16)
# np.savetxt(X=s_labels, fname="/Users/mpiekenbrock/pbsig/data/triangle_labels_500.txt", fmt="%d")

# %% Small data set (colex)
np.random.seed(1234)
X = np.random.uniform(size=(10,2))
R = rips_complex(X, radius=0.20)
R.expand(2)
er = rank_combs(faces(R, 1), n=X.shape[0], order="colex")
tr = rank_combs(faces(R, 2), n=X.shape[0], order="colex")
np.savetxt(X=er, fname="/Users/mpiekenbrock/pbsig/data/edge_ranks_colex_10.txt", fmt="%d")
np.savetxt(X=tr, fname="/Users/mpiekenbrock/pbsig/data/triangle_ranks_colex_10.txt", fmt="%d")



# %% Small data set (lex)
np.random.seed(1234)
X = np.random.uniform(size=(10,2))
R = rips_complex(X, radius=0.20)
R.expand(2)
er = rank_combs(faces(R, 1), n=X.shape[0], order="lex")
tr = rank_combs(faces(R, 2), n=X.shape[0], order="lex")
np.savetxt(X=er, fname="/Users/mpiekenbrock/pbsig/data/edge_ranks_lex_10.txt", fmt="%d")
np.savetxt(X=tr, fname="/Users/mpiekenbrock/pbsig/data/triangle_ranks_lex_10.txt", fmt="%d")



