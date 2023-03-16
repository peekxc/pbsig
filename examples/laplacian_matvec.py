from pbsig.linalg import *
from pbsig.vis import *
from splex import *
from itertools import combinations
from bokeh.plotting import show, figure
from bokeh.models import Range1d
from bokeh.io import output_notebook
output_notebook()

bokeh.__version__

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


## barbell chain
from bokeh.io import export_svg
from bokeh.models import Label, Text, ColumnDataSource
S = simplicial_complex([[0,1,2],[2,3,4]])
p = figure_complex(S, pos=np.array([[-1,-1],[-1,1],[0,0],[1,1],[1,-1]]), toolbar_location=None)
p.output_backend = "svg"
source = ColumnDataSource(dict(x=[-1,-1,0,1,1], y=[-1,1,0,1,-1], text=["a", "b", "c", "d", "e"]))
p.text(x='x',y='y',text='text', x_offset=10.80, y_offset=10.50, source=source)
p.x_range = Range1d(-1.5, 1.5)
p.y_range = Range1d(-1.5, 1.5)
p.xgrid.visible = False
p.ygrid.visible = False
p.axis.visible = False
show(p)
# export_svg(p, filename="/Users/mpiekenbrock/pbsig/notes/presentation/bowtie.svg")
K = filtration(enumerate(S))
D = boundary_matrix(K, p=2).todense()


from pbsig.persistence import ph 
from pbsig.betti import * 
dgms = ph(K, engine="dionysus")
fd = dict(zip(K.values(), K.keys()))

mu_query(K, lambda s: fd[s], p=1, R=(6,10,10,12)) # 2
mu_query(K, lambda s: fd[s], p=1, R=(7,10,11,12)) # 1
mu_query(K, lambda s: fd[s], p=1, R=(7,9,11,12)) # 0

p = figure_dgm(dgms[1], toolbar_location=None)
p.output_backend = "svg"
p.x_range = Range1d(5,13)
p.y_range = Range1d(5,13)
# i,j,k,l = 6,10,10,12
# i,j,k,l = 7,10,11,12
# i,j,k,l = 7,10,11,12
#p.rect(x=i+(j-i)/2,y=k+(l-k)/2,width=j-i,height=(l-k),line_alpha=0,fill_color="orange", fill_alpha=0.10)
# p.line([i,j],[k,k],line_dash="dotted", line_width=2, line_color="black")
# p.line([i,j],[l,l],line_dash="solid", line_width=2, line_color="black")
# p.line([i,i],[k,l],line_dash="dotted", line_width=2, line_color="black")
# p.line([j,j],[k,l],line_dash="solid", line_width=2, line_color="black")
p.scatter(dgms[1]['birth'], dgms[1]['death'], color="red", size=10)
# p.title = r"$$ \mu_1^R(K) = , \quad R = [7,10] \times [11,12] $$" 
# p.title = r"$$ \mu_1^R(K) = 2, \quad R = [6,10] \times [10,12] $$" 
# p.title = r"$$ \mu_1^R(K) = , \quad R = [7,10] \times [11,12] $$" 
show(p)

from bokeh.io import export_png, export_svg
# export_png(p, filename="/Users/mpiekenbrock/pbsig/notes/presentation/ex_mu1.png")
export_svg(p, filename="/Users/mpiekenbrock/pbsig/notes/presentation/ex_dgm.svg")





## matrix
p = figure(width=200,height=600)
p.x_range = Range1d(0,2)
p.y_range = Range1d(0,6)
source = ColumnDataSource(data=dict(x=D.nonzero()[1], y=5-D.nonzero()[0], text=[f"{i:2d}" for i in D[D.nonzero()]]))
p.text(x='x',y='y',text='text', x_offset=0, y_offset=-15, text_font_size='62px',source=source)
p.grid.grid_line_color = "black"
p.grid.ticker = list(range(6))
show(p)




# mu_query_mat(K, lambda s: fd[s], p=1, R=(6,10,10,12)) # 2

