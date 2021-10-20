#!/usr/bin/env python

from popex.popex_objects import Problem, CatParam
from popex import algorithm
import geostat
import forward



def main():
    deesse_simulator = geostat.DeesseSimulator()
    flow_solver = forward.FlowSolver(path_results = 'modflow')
    
    problem = Problem(generate_m=deesse_simulator.generate_m, # model generation function
                      compute_log_p_lik=flow_solver.compute_log_p_lik, # log likelihood (forward)
                      get_hd_pri=deesse_simulator.get_hd_pri) # prior hard conditioning

    algorithm.run_popex_mp(pb=problem,
                           path_res='popex',
                           path_q_cat=None,
                           nmp=40,
                           nmax=1000,
                           ncmax=(10,),
                           n_prior=1000)

if __name__=='__main__':
    main()
