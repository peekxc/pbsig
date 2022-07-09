import cppimport.import_hook


import SparseMatrix

assert 'PspFloatMatrix' in dir(SparseMatrix)


S = SparseMatrix.PspFloatMatrix(10, 10)
assert S.nnz == 0
assert "add_cols" in dir(S)


# SparseMatrix.print_matrix(S)