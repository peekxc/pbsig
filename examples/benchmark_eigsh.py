from scipy.sparse.linalg import eigsh
import timeit

x = np.random.uniform(size=L.shape[0])
timeit.timeit(lambda: L @ x, number=100)

timeit.timeit(lambda: eigsh(L, k=35, tol=1e-6, return_eigenvectors=False), number=100)


import primme
np.linalg.eigvalsh(LM)
primme.eigsh(L, k=15, maxiter=1500, ncv=5, tol=1e-6, method="PRIMME_Arnoldi",return_eigenvectors=False, return_unconverged=True, return_stats=True, printLevel=3)
timeit.timeit(lambda: np.linalg.eigvalsh(LM),  number=100)
timeit.timeit(lambda: eigsh(LS, k=35, maxiter=1500, tol=1e-6,return_eigenvectors=False),  number=100)
timeit.timeit(lambda: eigsh(L, k=35, maxiter=1500, tol=1e-6,return_eigenvectors=False), number=100)
timeit.timeit(lambda: primme.eigsh(L, k=35, maxiter=1500, tol=1e-6, method="PRIMME_Arnoldi",return_eigenvectors=False, return_unconverged=True), number=100)
