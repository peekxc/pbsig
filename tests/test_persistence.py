import numpy as np 
from pbsig.persistence import * 
from pbsig.simplicial import * 



def test_persistence():
  X = np.random.uniform(size=(20,2))
  K = rips_filtration(X, radius=0.25)
  dgm = barcodes(K)