import os
import shutil
import pickle

import flopy
import numpy as np

class FlowSolver():
    def __init__(self, path_results = 'modflow-obs'):
        self.measurements = np.array([ 0.89158034,  0.84550254, -0.14501171, -0.36723849,  0.11523723,
        0.1960391 ,  0.11089877, -0.09777393, -0.31272197, -1.23289254,
        0.59171852,  0.78778613,  0.86994152])
        self.locations = [ [ 16,  16], [ 32,  32], [ 48,  48], [ 80,  80],[ 96,  96], [112, 112], [ 16, 112], [ 32,  96], [ 48,  80], [ 63,  63], [ 80,  48], [ 96,  32], [112,  16]]
        self.err = 0.07
        self.path_results = path_results

    def run(self, img, name):
        k = self.img_to_k(img)
        mf6exe = 'mf6beta'
        workspace = self.workspace(name)
        nlay, nrow, ncol = 1, *np.shape(k)
        delr = 1
        delc = 1
        top = 1.
        botm = 0.

        pumping_welr, pumping_welc = 63, 63


        #initial condition
        strt = 1.
        #steady-state
        nper = 1
        perlen = [1]
        nstp = [1]
        tsmult = [1.]
        # ims solver parameters 
        outer_hclose = 1e-10
        outer_maximum = 100
        inner_hclose = 1e-10
        inner_maximum = 300
        rcloserecord = 1e-6
        linear_acceleration_gwf = 'CG'
        relaxation_factor = 1.0

        # build MODFLOW 6 files
        sim = flopy.mf6.MFSimulation(sim_name=name, version='mf6beta',
                                     exe_name=mf6exe,
                                     sim_ws=workspace)

        # create temporal discretization
        tdis_rc = []
        for i in range(nper):
            tdis_rc.append((perlen[i], nstp[i], tsmult[i]))
        tdis = flopy.mf6.ModflowTdis(sim, time_units='seconds',
                                     nper=nper, perioddata=tdis_rc)
        # create groundwater flow model and attach to simulation (sim)
        gwfname = 'gwf_' + name
        gwf = flopy.mf6.MFModel(sim, model_type='gwf6', modelname=gwfname, exe_name='mf6beta')

        # create iterative model solution
        imsgwf = flopy.mf6.ModflowIms(sim,
                                      print_option='summary',
                                      outer_hclose=outer_hclose,
                                      outer_maximum=outer_maximum,
                                      inner_maximum=inner_maximum,
                                      inner_hclose=inner_hclose,
                                      linear_acceleration=linear_acceleration_gwf,
                                      relaxation_factor=relaxation_factor,
                                      filename='{}.ims'.format(gwfname))
        sim.register_ims_package(imsgwf, [gwf.name])


        dis = flopy.mf6.ModflowGwfdis(gwf,
                                      nlay=nlay, nrow=nrow, ncol=ncol,
                                      delr=delr, delc=delc,
                                      top=top, botm=botm,)

        # node property flow
        npf = flopy.mf6.ModflowGwfnpf(gwf, k=k, save_specific_discharge=True, save_flows=True,)

        # boundary conditions
        chdlist = [[(0, i, 0), 1] for i in range(1, nrow-1)]
        chdlist += [[(0, j, ncol-1), 0.] for j in range(1, nrow-1)]
        #chdlist += [[(0, 0 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        #chdlist += [[(0, nrow-1 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]

        chd = flopy.mf6.ModflowGwfchd(gwf,
                                      stress_period_data=chdlist,
                                      save_flows=False,)

        # initial conditions
        ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

        #pumping
        wel = flopy.mf6.ModflowGwfwel(gwf,
                                          stress_period_data={0: [[(0, pumping_welr, pumping_welc), -0.003]]},
                                          pname='WEL')

        sto = flopy.mf6.ModflowGwfsto(gwf, steady_state={0:True})


        # output control
        oc = flopy.mf6.ModflowGwfoc(gwf,
                                    head_filerecord='{}.hds'.format(gwfname),
                                    budget_filerecord='{}.cbc'.format(gwfname),
                                    saverecord=[('HEAD', 'ALL')],)

        sim.write_simulation()
        sim.run_simulation()

        fpth = os.path.join(workspace,f'gwf_{name}.hds')
        hdobj = flopy.utils.HeadFile(fpth)
        return hdobj.get_alldata()
    
    def img_to_k(self, img):
        k = np.array(img, dtype='float')
        k[k==1] = 1e-2
        k[k==0] = 1e-4
        return k
    
    def compute_log_p_lik(self, model, imod):
        # prepare input
        img = model[0].param_val.reshape(128, 128)
        name = f'flow-{imod}'

        # run modflow
        heads = self.run(img=img, name=name)
        observations = np.array([heads[0,0,loc[0], loc[1]] for loc in self.locations])
        
        # save observations
        with open(f'{self.path_results}/observation-{imod}.pickle', 'wb') as file_handle:
            pickle.dump(observations, file_handle)

        # compute likelihood
        likelihood = np.sum(-0.5 * ((self.measurements - observations)**2) / (self.err**2))

        # clean-up
        shutil.rmtree(self.workspace(name))
        
        return likelihood
    
    def workspace(self, name):
        return f'modflow/{name}'
