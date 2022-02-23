#!/usr/bin/env python
import numpy as np

# 1.a
v = np.zeros(10)
v[4] = 1
print(v)

# 1.b
v = np.arange(10,50)
print(v)

# 1.c
v = v[::-1]
print(v)

# 1.d
v = np.arange(9).reshape(3,3)
print(v)

# 1.e
v = np.array([1,2,0,0,4,0])
nonzero_indices = v.nonzero()
print(v)
print(nonzero_indices)

# 1.f
v = np.random.random(30)
print(v.mean())

# 1.g
v = np.zeros((5,5))
v[0,:] = 1
v[:,0] = 1
v[-1,:] = 1
v[:,-1] = 1

print(v)

# 1.h
v =  np.zeros((8,8))
v[1::2,::2] = 1
v[::2,1::2] = 1
print(v)

# 1.i
v = np.tile([[1.,0.],[0.,1.]],(4,4))
print(v)

# 1.j
Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)

# 1.k
Z = np.random.random(10)
Z.sort()
print(Z)

# 1.l
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.array_equal(A,B)
print(equal)

# 1.m
Z = np.arange(10, dtype=np.int32)
print(Z.dtype)
Z = Z.astype('float32')
print(Z.dtype)

# 1.n
A = np.arange(9).reshape(3,3)
B = A + 1
C = np.dot(A,B)
D = np.diag(C)
print(D)
