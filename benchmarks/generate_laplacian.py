from splex import *

# %% Large data set 
np.random.seed(1234)
X = np.random.uniform(size=(500,2))
R = rips_complex(X, radius=0.10)
R.expand(2)
er = rank_combs(faces(R, 1), n=X.shape[0], order="colex")
tr = rank_combs(faces(R, 2), n=X.shape[0], order="colex")
np.savetxt(X=er, fname="/Users/mpiekenbrock/pbsig/data/edge_ranks_colex_500.txt", fmt="%d")
np.savetxt(X=tr, fname="/Users/mpiekenbrock/pbsig/data/triangle_ranks_colex_500.txt", fmt="%d")


# %% Small data set 
np.random.seed(1234)
X = np.random.uniform(size=(10,2))
R = rips_complex(X, radius=0.20)
R.expand(2)
er = rank_combs(faces(R, 1), n=X.shape[0], order="colex")
tr = rank_combs(faces(R, 2), n=X.shape[0], order="colex")
np.savetxt(X=er, fname="/Users/mpiekenbrock/pbsig/data/edge_ranks_colex_10.txt", fmt="%d")
np.savetxt(X=tr, fname="/Users/mpiekenbrock/pbsig/data/triangle_ranks_colex_10.txt", fmt="%d")

rank_colex