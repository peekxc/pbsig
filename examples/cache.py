import numpy as np
from collections import Counter

w26 = np.loadtxt('/Users/mpiekenbrock/Downloads/w26.txt')

accesses = Counter(w26)
import bokeh 

from bokeh.plotting import figure, show 
from bokeh.io import output_notebook
output_notebook()

from pbsig.color import bin_color


from scipy.interpolate import InterpolatedUnivariateSpline
f = InterpolatedUnivariateSpline(np.arange(len(w26)), w26)
from pbsig.persistence import sliding_window
sw = sliding_window(f, bounds=(0, len(w26)))
emb = sw(1000, d=120, L=10)

access_times = np.array([accesses[w] for w in w26])

from pbsig.linalg import cmds
from scipy.spatial.distance import squareform, pdist
Z = cmds(squareform(pdist(emb))**2, d=2)

from pbsig.persistence import sw_parameters
d,tau = sw_parameters(bounds=(0, len(w26)), d=120, L=10)


d = 120
w = int(len(w26)*d/(10*(d+1)))

len(w26)//w
p = figure(width=300, height=300)
p.line(Z[:,0], Z[:,1], color=bin_color())
show(p)

