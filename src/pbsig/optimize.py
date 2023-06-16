import numpy as np
from splex import *
from .linalg import * 
from typing import * 
from itertools import * 


def gradient_descent(
    fun: Callable, x0: Any, args: tuple = (), jac: Union[bool, Callable] = None, 
    bounds: List[tuple] = None, 
    tol: float = None
  ) -> dict:
  
  pass