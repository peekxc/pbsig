from pyflubber import interpolate
import numpy as np 


# svg_document.addElement(pysvg.text.text("Hello World"
T = np.array([[0,0], [1,0], [0.5, 1]])
R = np.array([[0,0], [1,0], [1,1], [0,1]])


import matplotlib.pyplot as plt

for i in np.linspace(0.0, 1.0, 10):
  S = interpolate(T, R, i, closed=False)
  plt.plot(*np.vstack((S, S[0,:])).T)



middle = interpolate(T, R, 0.5, closed=False)


import js2py
res, flubber = js2py.run_file("../js/flubber.min.js")

result= tempfile.sayHello("Stack Vidhya Reader");
var triangle = "M1,0 L2,2 L0,2 Z",
    pentagon = [[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]];

var interpolator = flubber.interpolate(triangle, pentagon);


