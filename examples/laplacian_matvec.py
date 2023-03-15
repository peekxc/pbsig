#pythran export laplacian_matvec(float[], int list, float[], int:int dict, float[], float[], float[], float[], int[])

from itertools import combinations
# x: ArrayLike, simplices: list, degree: ArrayLike, index: dict, v: ArrayLike, wfl: ArrayLike,  wfr: ArrayLike, ws: ArrayLike, sgn_pattern: tuple
def laplacian_matvec(x, simplices, degree, index, v, wfl,  wfr, ws, sgn_pattern):
  v.fill(0)
  v += degree * x.reshape(-1)
  for s_ind, s in enumerate(simplices):
    s_boundary = combinations(s, len(s)-1)
    for (f1, f2), sgn_ij in zip(combinations(s_boundary, 2), sgn_pattern):
      ii, jj = index[f1], index[f2]
      v[ii] += sgn_ij * x[jj] * wfl[ii] * ws[s_ind] * wfr[jj]
      v[jj] += sgn_ij * x[ii] * wfl[jj] * ws[s_ind] * wfr[ii]
  return v


from pbsig.vis import figure_complex
from splex import simplicial_complex
from bokeh.plotting import show, figure

## barbell chain 
S = simplicial_complex([[0,1,2],[2,3],[3,4],[4,5,6]])
show(figure_complex(S))

# from pbsig.

boundary_matrix
