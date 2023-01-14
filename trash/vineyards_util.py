n = 100
x0, y0 = np.repeat(0.0, n), np.random.uniform(size=n)
x1, y1 = np.repeat(1.0, n), np.random.uniform(size=n)
#a,b,c,d = x0,x1,y0,y1
#L = (c-a)/(b+d+c-a)

ind = np.argsort(y0)
P, Q = np.c_[x0,y0][ind,:], np.c_[x1,y1][ind,:]


def line_intersection(line1, line2):
  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
  def det(a, b): return a[0] * b[1] - a[1] * b[0]
  div = det(xdiff, ydiff)
  if div == 0:
    return(None)
  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div
  return x, y

from itertools import combinations 
n = P.shape[0]
results = []
for i,j in combinations(range(n), 2):
  pq = line_intersection((P[i,:], Q[i,:]), (P[j,:], Q[j,:]))
  if not(pq is None):
    results.append((i,j,pq[0],pq[1]))
results = np.array(results)
results = results[np.logical_and(results[:,2] >= 0.0, results[:,2] <= 1.0),:]
results = results[np.argsort(results[:,2]),:]
transpositions = results[:,:2].astype(int)

pp = np.array(range(n), dtype=int)
qp = np.argsort(Q[:,1])

import matplotlib.pyplot as plt 
plt.scatter(*P.T)
plt.scatter(*Q.T)
for i,j,k,l in zip(x0,x1,y0,y1):
  plt.plot((i,j), (k,l), c='black')
plt.scatter(*results[:,2:].T, zorder=30, s=5.5)
for x,y,label in zip(P[:,0],P[:,1],pp): plt.text(x, y, s=str(label),ha='left')
for x,y,label in zip(Q[qp,0],Q[qp,1],qp): plt.text(x, y, s=str(label),ha='right')


## Convert id transpositions to relative transpositions
rtransp = np.zeros(shape=transpositions.shape, dtype=int)
p_inv = np.argsort(pp)
perm = pp.copy()
for c,(i,j) in enumerate(transpositions):
  rtransp[c,:] = [p_inv[i],p_inv[j]]
  perm[[p_inv[j],p_inv[i]]] = perm[[p_inv[i],p_inv[j]]]
  # p_inv[[i,j]] = [j,i]
  p_inv = np.argsort(perm)
  
assert np.all(qp == perm)
assert np.all(np.abs(rtransp[:,0]-rtransp[:,1]) == 1)




# # S = ss.random(m=3500, n=3500, density=0.01, format='coo')

# def do_dim_ops(M, N: int = 1500, row: bool = False):
#   n, m = M.shape
#   if row:
#     I_, J_ = np.random.choice(range(n), N), np.random.choice(range(n), N)
#   else:
#     I_, J_ = np.random.choice(range(m), N), np.random.choice(range(m), N)
#   I = np.minimum(I_, J_)
#   J = np.minimum(I_, J_)
#   if not(row):
#     for (i, j) in zip(I, J):
#       M[:,j] = M[:,i] + M[:,j]
#   else:
#     for (i, j) in zip(I, J):
#       M[j,:] = M[i,:] + M[j,:]
#   return(M) 

# S_coo = S.tocoo()
# S_lil = S.tolil()
# S_csc = S.tocsc()
# S_dok = S.todok()
# S_csr = S.tocsr()
# S_dense = S_coo.A

# import timeit as timeit
# timeit.timeit(lambda: do_dim_ops(S_lil, 15), number=1)
# timeit.timeit(lambda: do_dim_ops(S_lil, 15, row=True), number=1)
# timeit.timeit(lambda: do_dim_ops(S_csc, 15, row=False), number=1)
# timeit.timeit(lambda: do_dim_ops(S_csr, 15, row=True), number=1)
# timeit.timeit(lambda: do_dim_ops(S_dok, 15, row=False), number=1)
# timeit.timeit(lambda: do_dim_ops(S_dense, 15, row=False), number=1)


# def do_permutations(M, N: int = 1500, row: bool = False):
#   ind = np.array(range(S.shape[0])) 
#   if not(row):
#     for i in range(N):
#       P = np.eye(S.shape[0])[:,np.random.permutation(ind)]
#       M = M @ P
#   else: 
#     for i in range(N):
#       P = np.eye(S.shape[0])[:,np.random.permutation(ind)]
#       M = P @ M
#   return(M)

# def do_permutations(M, N: int = 1500, row: bool = False):
#   ind = np.array(range(S.shape[0])) 
#   if not(row):
#     for i in range(N):
#       j = np.random.choice(range(S.shape[0]-1))
#       M[:,(j,j+1)] = M[:,(j+1,j)]
#   else: 
#     for i in range(N):
#       j = np.random.choice(range(S.shape[0]-1))
#       M[(j,j+1),:] = M[(j+1,j),:]
#   return(M)

# import timeit as timeit
# timeit.timeit(lambda: do_permutations(S_lil, 150), number=1)
# timeit.timeit(lambda: do_permutations(S_lil, 150, row=True), number=1)
# timeit.timeit(lambda: do_permutations(S_csc, 150, row=False), number=1)
# timeit.timeit(lambda: do_permutations(S_csc, 150, row=True), number=1)
# timeit.timeit(lambda: do_permutations(S_csr, 150, row=True), number=1)
# timeit.timeit(lambda: do_permutations(S_csr, 150, row=False), number=1)
# timeit.timeit(lambda: do_permutations(S_dok, 150, row=False), number=1)
# timeit.timeit(lambda: do_permutations(S_dense, 150, row=False), number=1)

# from line_profiler import LineProfiler 
# profiler = LineProfiler()
# profiler.add_function(do_column_ops)
# profiler.enable_by_count()
# do_column_ops(S_lil, 1500)
# profiler.print_stats(output_unit=1e-3)

# Try Cython? 
