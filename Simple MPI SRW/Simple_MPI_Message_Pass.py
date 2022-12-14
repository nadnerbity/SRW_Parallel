#!/usr/local/python
#

# This file is for testing how to build an MPI simulation using SRW. It uses
# the python file Simple_ER_Sim.py to define an SRW simulation to have a
# non-trival computation to do on multiple processors.

from mpi4py import MPI
import time
import Simple_ER_Sim

# Open up the MPI communication. If you run this without mpiexec you'll just
# get one processor.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = comm.Get_size()

nx = 2**10

# Hold here and wait for all processors to catch up.
comm.Barrier()

# Run the N simulations and time how long each takes and report that back.
start = time.time()
# Create an array on the head node
if rank == 0:
    print('The world size is', world, 'processors.')

print('I am', rank, 'and I am running a simulation of size nx = ', nx)
start = time.time()
wfr = Simple_ER_Sim.run_simple_er_simulation(nx)
print('Execution time of processor', rank, 'was', time.time() - start)

# Hold here and wait for all processors to catch up.
comm.Barrier()

# Copy the N-1 simulations to the head node.
if rank != 0:
    comm.send(wfr, dest=0, tag=rank)
elif rank == 0:
    # Collect the data from the other nodes
    start = time.time()
    for i in range(1, world):
        # print('in the receive loop')
        temp = comm.recv(source=i, tag=i)


    print('Data collection time', time.time() - start)