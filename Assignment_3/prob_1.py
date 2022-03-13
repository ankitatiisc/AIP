import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

import pdb

def numerator_part_1(x,y,sigma_x, sq_sigma_z):
    #K = (1./(2*sigma_x)) * (1./sp.sqrt(2*math.pi*sq_sigma_z))
    #return  x * sp.exp(-((-x/sigma_x) + ((y-x)**2)/(2 * sq_sigma_z)))
    return  x * sp.exp((x * (2*sq_sigma_z - sigma_x*x + 2*y*sigma_x))/(2 * sigma_x * sq_sigma_z ) )

def numerator_part_2(x,y,sigma_x, sq_sigma_z):
    #K = (1./(2*sigma_x)) * (1./sp.sqrt(2*math.pi*sq_sigma_z))
    #return  x * sp.exp(-((x/sigma_x) + ((y-x)**2)/(2 * sq_sigma_z)))
    return  x * sp.exp((x * (-2*sq_sigma_z - sigma_x*x + 2*y*sigma_x))/(2 * sigma_x * sq_sigma_z ) )

def denominator_part_1(x,y,sigma_x, sq_sigma_z):
    #K = (1./(2*sigma_x)) * (1./sp.sqrt(2*math.pi*sq_sigma_z))
    #return sp.exp(-((-x/sigma_x) + ((y-x)**2)/(2 * sq_sigma_z)))
    return  sp.exp((x * (2*sq_sigma_z - sigma_x*x + 2*y*sigma_x))/(2 * sigma_x * sq_sigma_z ) )

def denominator_part_2(x,y,sigma_x, sq_sigma_z):
    #K = (1./(2*sigma_x)) * (1./sp.sqrt(2*math.pi*sq_sigma_z))
    #return  sp.exp(-((x/sigma_x) + ((y-x)**2)/(2 * sq_sigma_z)))
    return  sp.exp((x * (-2*sq_sigma_z - sigma_x*x + 2*y*sigma_x))/(2 * sigma_x * sq_sigma_z ) )

def num_full(x,y,sigma_x,sq_sigma_z):
    return x * sp.exp(-sp.Abs(x)/sigma_x) * sp.exp(((y-x)**2)/(2*sq_sigma_z))

def denom_full(x,y,sigma_x,sq_sigma_z):
    return sp.exp(-sp.Abs(x)/sigma_x) * sp.exp(((y-x)**2)/(2*sq_sigma_z))
x_estimate = []
y_val = []
sigma_x = 1
sq_sigma_z = 0.1
for y in np.arange(-2,2.1,0.1):        
    x = sp.Symbol("x")
    num_a = sp.integrate(numerator_part_1(x,y,sigma_x,sq_sigma_z), (x, -sp.oo, 0))
    num_b = sp.integrate(numerator_part_2(x,y,sigma_x,sq_sigma_z), (x, 0, sp.oo))
    den_a = sp.integrate(denominator_part_1(x,y,sigma_x,sq_sigma_z), (x, -sp.oo, 0))
    den_b = sp.integrate(denominator_part_2(x,y,sigma_x,sq_sigma_z), (x, 0, sp.oo))
    y_val.append(y)
    x_estimate.append((num_a.evalf() + num_b.evalf())/(den_a.evalf() + den_b.evalf()))
    #num_f = sp.integrate(num_full(x,y,sigma_x,sq_sigma_z), (x, -sp.oo,sp.oo))
    #den_f = sp.integrate(denom_full(x,y,sigma_x,sq_sigma_z), (x, -sp.oo,sp.oo))
    #x_estimate.append(num_f.evalf()/den_f.evalf())
    print(y, x_estimate[-1])
    #pdb.set_trace()
    #print(num_a.evalf())
    #print(num_b.evalf())
    #print(den_a.evalf())
    #print(den_b.evalf())
    #print((num_a.evalf() + num_b.evalf())/(den_a.evalf() + den_b.evalf()))

plt.plot(y_val,x_estimate,color='g',label='MMSE Estimate for Laplacian')
plt.plot(y_val,y_val,color='r',label='Identity')
plt.legend()
plt.title('MMSE Estimate vs Y')
plt.xlabel('Y')
plt.ylabel('X_estimate')
plt.savefig('mmse_plot.png')
plt.close()