#!/usr/bin/env python

import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

randNum = numpy.zeros(1)

for i in range(1,5):
    if rank == i:
        randNum = numpy.random.random_sample(1)
        comm.send(randNum, dest=0)
    
if rank == 0:
    sum = 0
    for i in range(1,5):
        sum += comm.recv(source=i)
    print("The sum from the prossess %i is %f" % (rank,sum))