from splex import *
from more_itertools import collapse 

# %% 
from pbsig.perfect_hash import perfect_hash
perfect_hash(er, output = "expression", solver = "linprog", k_min = 3, k_max = 15, n_tries = 100, n_prime = 1000)
  
# %% Large data set (colex)
np.random.seed(1234)
X = np.random.uniform(size=(250,2))
R = rips_complex(X, radius=0.10)
R.expand(2)
er = rank_combs(faces(R, 1), n=X.shape[0], order="colex")
tr = rank_combs(faces(R, 2), n=X.shape[0], order="colex")
np.savetxt(X=er, fname="/Users/mpiekenbrock/pbsig/data/edge_ranks_colex_500.txt", fmt="%d")
np.savetxt(X=tr, fname="/Users/mpiekenbrock/pbsig/data/triangle_ranks_colex_500.txt", fmt="%d")
s_labels = np.fromiter(collapse(list(faces(R, 2))), dtype=np.uint16)
np.savetxt(X=s_labels, fname="/Users/mpiekenbrock/pbsig/data/triangle_labels_500.txt", fmt="%d")

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
