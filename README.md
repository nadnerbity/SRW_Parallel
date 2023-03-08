A series of python scripts to divide SRW simulations into subsets and run each piece independently. The idea is that sometimes a simulation is big enough that it makes sense to split it across multiple cores and do it quickly when prototyping input decks. The ultimate speed up is \sqrt(N_{cores}).

The current method splits up the simulation space, performs the simulation and then recombines the simulation on the head node. This is not optimal when you also wish to then use the physical optics component of SRW, which reduces mesh requirements by quite a bit.

The Physical Optics speed-up doesn't work because the MPI simulations do not transfer the radius of curvature (SRWLWfr.mesh.Rx and Ry) to the head node. For each part of the simulation they are all different, and I have yet to put time into figuring out the best way to combine them. I suspect the way forward is to perform the physical optics on each piece and then combine them, but that is a future update.

Layout of this repo:
