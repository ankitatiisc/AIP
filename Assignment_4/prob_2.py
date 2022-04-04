import copy
import numpy as np
from torch import solve
import sympy as sp

def numerator_left_part(x):
    return x * (1./6.) * sp.exp(x/3.)

def numerator_right_part(x):
    return x * (1./6.) * sp.exp(-x/3.)

def denominator_left_part(x):
    return (1./6.) * sp.exp(x/3.)

def denominator_right_part(x):
    return (1./6.) * sp.exp(-x/3.)

def get_numerator_integral(a,b):
    x = sp.Symbol("x")
    if a < 0 and b < 0:
        num = sp.integrate(numerator_left_part(x), (x, a, b))
    elif a < 0 and b >= 0.:
        num = sp.integrate(numerator_left_part(x), (x, a, 0)) + sp.integrate(numerator_right_part(x), (x, 0, b)) 
    elif a >= 0. and b >= 0.:
        num = sp.integrate(numerator_right_part(x), (x, a, b)) 
    return num.evalf()

def get_denominator_integral(a,b):
    x = sp.Symbol("x")
    if a < 0 and b < 0:
        num = sp.integrate(denominator_left_part(x), (x, a, b))
    elif a < 0 and b >= 0.:
        num = sp.integrate(denominator_left_part(x), (x, a, 0)) + sp.integrate(denominator_right_part(x), (x, 0, b)) 
    elif a >= 0. and b >= 0.:
        num = sp.integrate(denominator_right_part(x), (x, a, b)) 
    return num.evalf()

def distortion_left_part(x,t):
    return ((x-t)**2)*(1./6.) * sp.exp(x/3.)

def distortion_right_part(x,t):
    return ((x-t)**2)*(1./6.) * sp.exp(-x/3.)

def compute_distortion(X,U):
    distortion = 0.0
    x = sp.Symbol("x")
    num = sp.integrate(distortion_left_part(x,X[0]), (x, -sp.oo, U[0]))
    distortion += num.evalf()

    x = sp.Symbol("x")
    num = sp.integrate(distortion_left_part(x,X[1]), (x, U[0], U[1]))
    distortion += num.evalf()

    x = sp.Symbol("x")
    num = sp.integrate(distortion_right_part(x,X[2]), (x, U[1], U[2]))
    distortion += num.evalf()

    x = sp.Symbol("x")
    num = sp.integrate(distortion_right_part(x,X[3]), (x, U[2], sp.oo))
    distortion += num.evalf()

    return distortion

X = [-5, -2, 2, 5]
U = [0,0,0]
i = 0
#for i in range(100):
while True:
    print('-'*100+'\nIter {}'.format(i+1))
    u1 = (X[0] + X[1])/2.
    u2 = (X[1] + X[2])/2.
    u3 = (X[2] + X[3])/2.
    print(u1, u2, u3)
    X_prev = copy.deepcopy(X)
    X[0] = get_numerator_integral(-1000, u1)/get_denominator_integral(-1000, u1)
    X[1] = get_numerator_integral(u1, u2)/get_denominator_integral(u1, u2)
    X[2] = get_numerator_integral(u2, u3)/get_denominator_integral(u2, u3)
    X[3] = get_numerator_integral(u3, 1000)/get_denominator_integral(u3, 1000)
    change = 0.0
    for x_prev,x_curr in zip(X_prev,X):
        change += ( abs(x_prev - x_curr) )
    print('Change from previous values is ', change)
    i += 1
    print(X)
    U = [u1, u2, u3]
    if change <= float(4e-6):
        break

print('Final decision points are : \n',X)
print('Lloyd Max distortiion is ', compute_distortion(X, U))

#part 1
delta = 4.613416
X = [(-3*delta)/2., (-delta)/2., (delta)/2., (3*delta)/2. ]
U = [-delta, 0., delta]

print('PART 1')
print(X)
print(U)
print('distortiion is ', compute_distortion(X, U))
