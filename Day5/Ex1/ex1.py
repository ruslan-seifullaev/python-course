#!/usr/bin/env python

import numpy as np

#1. Scipy
import scipy.linalg as la 

#   Linear Algebra

# a.
#A = np.array([(1, 2, 3),(4, 5, 6),(7, 8, 9)]) #ill-conditioned
A = np.array([(1, 2, 3),(4, 0, 6),(7, 8, 9)])
          
# b.
b = np.array([1,2,3])

# c.
x = la.solve(A,b)
print(x)

# d.
print((np.dot(A,x) - b))

# e.
B = np.random.random(9).reshape((3,3))
x_rand = la.solve(A,B)
print(x_rand)
print((np.dot(A,x_rand) - B))

# f.
lambda_i,v_i = la.eig(A)
print('Eigenvalues:', lambda_i)  
print('Eigenvectors:', v_i)

# g.
A_inv = la.inv(A)
det_A = la.det(A)
print('A_inv =',A_inv)
print('det A = ', det_A)

# h.
for order in ['fro', np.inf, 1, 2]: 
    print('Norm of A with order %s: %s' % (order, la.norm(A, ord = order)))
    
#   Statistics
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt 

def plot_disctibution(data, ax_left, ax_right, bins_N, title):
     fig = plt.figure()
     ax  = fig.add_subplot(111)
     ax.hist(data, range=(ax_left,ax_right), bins=bins_N, density=True, stacked=True)
     ax.set_title(title)
     plt.show()
     
# a.
mu = 1
data_poisson = poisson.rvs(mu, size=1000)
plot_disctibution(data_poisson, 0, 5, 10, 'Poisson')

# b.
mu = 0
sigma = 1
data_gaussian = norm.rvs(mu, sigma, size=1000)
plot_disctibution(data_gaussian, -5, 5, 20, 'Gaussian')

# c.
from scipy import stats
print(stats.ttest_ind(data_poisson,data_gaussian))