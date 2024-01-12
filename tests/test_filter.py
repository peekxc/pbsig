import numpy as np 
from pbsig.interpolate import ParameterizedFilter
import splex as sx

S = sx.simplicial_complex([range(5)])
p_filter = ParameterizedFilter(S)
p_filter(0.1)

vals1 = np.random.uniform(size=len(S))
vals2 = np.random.uniform(size=len(S))

interp = lambda alpha: (1 - alpha) * vals1 + alpha * vals2
p_filter.family = interp
p_filter(0.1)

p_filter.family = [interp(a) for a in np.linspace(0, 1, 10)]

p_filter()
