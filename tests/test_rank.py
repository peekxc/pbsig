## Tests various closed-form expressions for Laplacian matrices
import numpy as np 
from itertools import combinations
from scipy.sparse import diags, spmatrix
from pbsig.persistence import boundary_matrix
from pbsig.simplicial import *
from pbsig.utility import *
from pbsig.linalg import *
from splex import simplicial_complex
from splex import ComplexLike


from pbsig.itertools import rank_tuple, unrank_tuple
def test_tuple_ranking():
  N = tuple(np.random.choice(range(10), size=3))
  assert all([rank_tuple(unrank_tuple(i, N), N) == i for i in range(prod(N))])