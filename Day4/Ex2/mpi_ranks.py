#!/usr/bin/env python

from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.Get_rank()
print("hello world from process ", rank)