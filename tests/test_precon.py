import numpy as np 
import networkx as nx
from scipy.sparse.linalg import eigsh
from pbsig.simplicial import graph_laplacian, complete_graph, is_symmetric, edge_iterator
from pbsig.precon import graph_sparsifier

## Check pathological graphs
K = complete_graph(30)
L = graph_laplacian(K, normalize=True)
ew = eigsh(L, k=L.shape[0]-1, return_eigenvectors=False)

## Compute sample graph
G = nx.connected_watts_strogatz_graph(100, k=10, p=0.30)
LG = graph_laplacian(G, normalize=True)
ew_g = eigsh(LG, k=LG.shape[0]-1, return_eigenvectors=False)

## Construct spectral sparsifier
H = graph_sparsifier(LG)
ew_h = eigsh(graph_laplacian(H, normalize=True), k=LG.shape[0]-1, return_eigenvectors=False)
# ew_h *= max(ew_g)/max(ew_h)
np.sort((1-(1/np.sqrt(L.shape[0])))*ew_g)
np.sort(ew_h)
np.sort(ew_g)
str(H), str(G)

## Test Lanczos w/ preconditioner
import primme
from sksparse.cholmod import cholesky, cholesky_AAt 

default_opts = { }
primme.eigsh(A, 
)

#L = graph_laplacian((K['edges'], len(K['vertices'])))
# int((L.nnz - L.shape[0])/2)