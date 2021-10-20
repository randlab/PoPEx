#!/bin/env python

from popex.popex_objects import Problem
from popex import algorithm
from fluvial import FluvialSimulation
from tracer import TracerTest

fluvial_simulation = FluvialSimulation(nthreads=2)
tracer_test = TracerTest(steps_factor=4,
                         working_dir='modflow',
                         modpath='steady-state')

def main():
    problem = Problem(generate_m=fluvial_simulation.generate_m,
                      compute_log_p_lik=tracer_test.compute_log_p_lik,
                      get_hd_pri=fluvial_simulation.get_hd_pri)

    algorithm.run_popex_mp(pb=problem,
                           path_res='popex/',
                           path_q_cat='./',
                           nmp=32,
                           nmax=4000,
                           ncmax=(10,),
                           n_prior=0)

if __name__=='__main__':
    main()
