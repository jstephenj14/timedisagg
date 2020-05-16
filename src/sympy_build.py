
import sympy as sym

from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
import numpy as np
from scipy.linalg import toeplitz
import scipy.linalg as linalg


x, y = sym.symbols('x, y')
A = sym.Matrix([[x-2, 0], [0, x-3]])
func = A.det().as_poly().as_expr()
my_func = lambdify((x),func)
results = minimize(my_func,[0.1])



from scipy import optimize
def f(x):
    deter_A = A.det()
    return deter_A.as_poly().as_expr()

result = optimize.minimize_scalar(f)


a, b, G = sympy.symbols('a b G')
func = (G - a) ** 2 + b
my_func = lambdify((G, a, b), -1 * func)
results = minimize(my_func, [0.1, 0.1, 0.1])