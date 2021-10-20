import pickle
import os
import sys
import shutil
import numpy as np
import pandas as pd
import flopy

class TracerTest():
    def __init__(self, cell_divisions=5, steps_factor=4, alphal=4.,
                 working_dir='./', reference_run=False, modpath='steady-state'):
        self.nx = 100
        self.ny = 100
        self.cell_divisions = cell_divisions
        self.steps_factor = steps_factor
        self.alphal = alphal
        self.working_dir = working_dir
        self.modpath = modpath
        # ensure reference values are here
        if reference_run is False:
            with open(f'measurement-{steps_factor}.pickle', 'rb') as file_handle:
                self.measurement = pickle.load(file_handle)
  
    def run(self, img, name):
        
        mf6exe = 'mf6'
        workspace = self.modflow_workspace(name)
        k = self.get_k(img)
        porosity = self.get_porosity(img)
        ss = self.get_specificstorage(img)
        k_new, porosity_new, ss_new = self.scale(k=k, porosity=porosity, ss=ss, n=self.cell_divisions)

        # spatial discretization
        nlay, nrow, ncol = 1, *np.shape(k_new)
        delr = 500. / nrow
        delc = 500. / ncol
        top = 10.
        botm = 0.
        
        pumping_welr, pumping_welc = self.pumping_well(n=self.cell_divisions)
        injection_welr, injection_welc = self.injection_well(n=self.cell_divisions)

        
        #initial condition
        strt = 10.
        # time discretization: 4 periods - 2 steady-state + 2 transient
        nper = 4
        perlen = [1, 1, 3600, 1728000.]
        nstp = [1, 1, 36, 60*self.steps_factor]
        tsmult = [1., 1., 1., 1.]
        # ims solver parameters 
        outer_hclose = 1e-10
        outer_maximum = 100
        inner_hclose = 1e-10
        inner_maximum = 300
        rcloserecord = 1e-6
        linear_acceleration_gwf = 'CG'
        linear_acceleration_gwt = 'BICGSTAB'
        #relaxation_factor = 0.97
        relaxation_factor = 1.0
        
        #hnoflo = 1e30
        #hdry = -1e30
        #hk = 0.0125
        #laytyp = 0
        diffc = 1e-9
        alphal = self.alphal
        alphath = 0.1 * self.alphal
        #alphatv = 0.006

        #nouter, ninner = 100, 300
        #hclose, rclose, relax = 1e-4, 1e-3, 0.97



        # build MODFLOW 6 files
        sim = flopy.mf6.MFSimulation(sim_name=name, version='mf6',
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
        gwf = flopy.mf6.MFModel(sim, model_type='gwf6', modelname=gwfname, exe_name='mf6')
        
        # create iterative model solution
        imsgwf = flopy.mf6.ModflowIms(sim,
                                      print_option='summary',
                                      outer_hclose=outer_hclose,
                                      outer_maximum=outer_maximum,
                                      inner_maximum=inner_maximum,
                                      inner_hclose=inner_hclose,
                                      rcloserecord=rcloserecord,
                                      linear_acceleration=linear_acceleration_gwf,
                                      relaxation_factor=relaxation_factor,
                                      filename='{}.ims'.format(gwfname))
        sim.register_ims_package(imsgwf, [gwf.name])
        

        dis = flopy.mf6.ModflowGwfdis(gwf,
                                      nlay=nlay, nrow=nrow, ncol=ncol,
                                      delr=delr, delc=delc,
                                      top=top, botm=botm,)

        # node property flow
        npf = flopy.mf6.ModflowGwfnpf(gwf, k=k_new, save_specific_discharge=True)

        # boundary conditions
        chdlist = [[(0, i, 0), 0.5] for i in range(1, nrow-1)]
        chdlist += [[(0, j, ncol-1), 0.] for j in range(1, nrow-1)]
        chdlist += [[(0, 0 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        chdlist += [[(0, nrow-1 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        
        chd = flopy.mf6.ModflowGwfchd(gwf,
                                      stress_period_data=chdlist,
                                      save_flows=False,)

        # initial conditions
        ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)
        
        #pumping
        wel = flopy.mf6.ModflowGwfwel(gwf,
                                          stress_period_data={1: [[(0, pumping_welr, pumping_welc), -0.07, 0]],
                                                          2: [[(0, pumping_welr, pumping_welc), -0.07, 0],[(0,injection_welr, injection_welc), 1/3600, 1]], 
                                                          3:[[(0, pumping_welr, pumping_welc), -0.07, 0]],},
                                      pname='WEL',
                                      auxiliary='CONCENTRATION')

        sto = flopy.mf6.ModflowGwfsto(gwf,
                                      #ss=1e-6,
                                      ss=ss_new,
                                      steady_state={0:True, 1:True}, transient={2:True, 3:True})


        # output control
        oc = flopy.mf6.ModflowGwfoc(gwf,
                                    head_filerecord='{}.hds'.format(gwfname),
                                    saverecord=[('head', 'last')],)

        
        # create gwt model
        gwtname = 'gwt_' + name
        gwt = flopy.mf6.MFModel(sim, model_type='gwt6', modelname=gwtname,)

        # create iterative model solution
        imsgwt = flopy.mf6.ModflowIms(sim,
                                      print_option='summary',
                                      outer_hclose=outer_hclose,
                                      outer_maximum=outer_maximum,
                                      inner_maximum=inner_maximum,
                                      inner_hclose=inner_hclose,
                                      rcloserecord=rcloserecord,
                                      linear_acceleration=linear_acceleration_gwt,
                                      relaxation_factor=relaxation_factor,
                                      filename='{}.ims'.format(gwtname))
        sim.register_ims_package(imsgwt, [gwt.name])
        
        dis = flopy.mf6.ModflowGwtdis(gwt, nlay=nlay, nrow=nrow, ncol=ncol,
                                      delr=delr, delc=delc,
                                      top=top, botm=botm, idomain=1)

        # initial conditions
        strt_gwt = np.zeros((nlay, nrow, ncol))
        ic = flopy.mf6.ModflowGwtic(gwt, strt=strt_gwt)

        # advection
        adv = flopy.mf6.ModflowGwtadv(gwt, scheme='TVD',
                                      filename='{}.adv'.format(gwtname))

        # dispersion
        dsp = flopy.mf6.ModflowGwtdsp(gwt, diffc=diffc,
                                     alh=alphal, alv=alphal,
                                     ath1=alphath,
                                     filename='{}.dsp'.format(gwtname))

        # storage
        stogwt = flopy.mf6.ModflowGwtmst(gwt, porosity=porosity_new,)

        # sources
        sourcerecarray = [('WEL', 'AUX', 'CONCENTRATION')]
        ssm = flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)

        # output control
        oc = flopy.mf6.ModflowGwtoc(gwt,
                                    budget_filerecord='{}.cbc'.format(gwtname),
                                    concentration_filerecord='{}.ucn'.format(gwtname),
                                    concentrationprintrecord=[
                                        ('COLUMNS', 10, 'WIDTH', 15,
                                         'DIGITS', 6, 'GENERAL')],
                                    saverecord=[('CONCENTRATION', 'ALL')],
                                    printrecord=[('CONCENTRATION', 'LAST'),
                                                 ('BUDGET', 'LAST')])

        # GWF GWT exchange
        gwfgwt = flopy.mf6.ModflowGwfgwt(sim, exgtype='GWF6-GWT6',
                                         exgmnamea=gwfname, exgmnameb=gwtname)


        
        sim.write_simulation()
        sim.run_simulation()
        
    def run_modpath_transient(self, img, name):
        mf6exe = 'mf6'
        workspace = self.modpath_workspace(name)
        
        k = self.get_k(img)
        porosity = self.get_porosity(img)
        ss = self.get_specificstorage(img)
        k_new, porosity_new, ss_new = self.scale(k=k, porosity=porosity, ss=ss, n=self.cell_divisions)

        # spatial discretization
        nlay, nrow, ncol = 1, *np.shape(k_new)
        delr = 500. / nrow
        delc = 500. / ncol
        top = 10.
        botm = 0.
        
        pumping_welr, pumping_welc = self.pumping_well(n=self.cell_divisions)
        injection_welr, injection_welc = self.injection_well(n=self.cell_divisions)

        
        #initial condition
        strt = 10.
        # time discretization: 4 periods - 2 steady-state + 2 transient
        nper = 2
        perlen = [3600, 864000.]
        nstp = [36, 30*self.steps_factor]
        tsmult = [1., 1.]
        # ims solver parameters 
        outer_hclose = 1e-10
        outer_maximum = 100
        inner_hclose = 1e-10
        inner_maximum = 300
        rcloserecord = 1e-6
        linear_acceleration_gwf = 'CG'
        #relaxation_factor = 0.97
        relaxation_factor = 1.0



        # build MODFLOW 6 files
        sim = flopy.mf6.MFSimulation(sim_name=name, version='mf6',
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
        gwf = flopy.mf6.MFModel(sim, model_type='gwf6', modelname=gwfname, exe_name='mf6')
        
        # create iterative model solution
        imsgwf = flopy.mf6.ModflowIms(sim,
                                      print_option='summary',
                                      outer_hclose=outer_hclose,
                                      outer_maximum=outer_maximum,
                                      inner_maximum=inner_maximum,
                                      inner_hclose=inner_hclose,
                                      rcloserecord=rcloserecord,
                                      linear_acceleration=linear_acceleration_gwf,
                                      relaxation_factor=relaxation_factor,
                                      filename='{}.ims'.format(gwfname))
        sim.register_ims_package(imsgwf, [gwf.name])
        

        dis = flopy.mf6.ModflowGwfdis(gwf,
                                      nlay=nlay, nrow=nrow, ncol=ncol,
                                      delr=delr, delc=delc,
                                      top=top, botm=botm,)

        # node property flow
        npf = flopy.mf6.ModflowGwfnpf(gwf, k=k_new, save_specific_discharge=True, save_flows=True)

        # boundary conditions
        chdlist = [[(0, i, 0), 0.5] for i in range(1, nrow-1)]
        chdlist += [[(0, j, ncol-1), 0.] for j in range(1, nrow-1)]
        chdlist += [[(0, 0 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        chdlist += [[(0, nrow-1 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        
        chd = flopy.mf6.ModflowGwfchd(gwf,
                                      stress_period_data=chdlist,
                                      save_flows=False,)

        # initial conditions
        ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)
        
        #pumping
        wel = flopy.mf6.ModflowGwfwel(gwf,
                                          stress_period_data={
                                                          0: [[(0, pumping_welr, pumping_welc), -0.07, 0],[(0,injection_welr, injection_welc), 1/3600, 1]], 
                                                          1:[[(0, pumping_welr, pumping_welc), -0.07, 0]],},
                                      pname='WEL',
                                      auxiliary='CONCENTRATION')

        sto = flopy.mf6.ModflowGwfsto(gwf,
                                      #ss=1e-6,
                                      ss=ss_new,
                                      transient={0:True, 1:True})


        # output control
        oc = flopy.mf6.ModflowGwfoc(gwf,
                                    head_filerecord='{}.hds'.format(gwfname),
                                    budget_filerecord='{}.cbc'.format(gwfname),
                                    saverecord=[('head', 'all'), ('budget', 'all')],)
       
        sim.write_simulation()
        sim.run_simulation()
        
        mp = flopy.modpath.Modpath7.create_mp7(modelname='modpath', trackdir='forward',
        flowmodel=gwf, model_ws=workspace,
        rowcelldivisions=1, columncelldivisions=1, layercelldivisions=1,
        exe_name="mp7")
        flopy.modpath.mp7bas.Modpath7Bas(mp, porosity=porosity_new, defaultiface=None, extension='mpbas')

        mp.write_input()
        mp.run_model()
        
    def run_modpath(self, img, name):
        mf6exe = 'mf6'
        workspace = self.modpath_workspace(name)
        k = self.get_k(img)
        porosity = self.get_porosity(img)
        ss = self.get_specificstorage(img)
        k_new, porosity_new, ss_new = self.scale(k=k, porosity=porosity, ss=ss, n=self.cell_divisions)
        # spatial discretization
        nlay, nrow, ncol = 1, *np.shape(k_new)
        delr = 500. / nrow
        delc = 500. / ncol
        top = 10.
        botm = 0.

        pumping_welr, pumping_welc = self.pumping_well(n=self.cell_divisions)
        injection_welr, injection_welc = self.injection_well(n=self.cell_divisions)

        
        #initial condition
        strt = 10.
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
        sim = flopy.mf6.MFSimulation(sim_name=name, version='mf6',
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
        gwf = flopy.mf6.MFModel(sim, model_type='gwf6', modelname=gwfname, exe_name='mf6')
        
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
        npf = flopy.mf6.ModflowGwfnpf(gwf, k=k_new, save_specific_discharge=True, save_flows=True,)

        # boundary conditions
        chdlist = [[(0, i, 0), 0.5] for i in range(1, nrow-1)]
        chdlist += [[(0, j, ncol-1), 0.] for j in range(1, nrow-1)]
        chdlist += [[(0, 0 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        chdlist += [[(0, nrow-1 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        
        chd = flopy.mf6.ModflowGwfchd(gwf,
                                      stress_period_data=chdlist,
                                      save_flows=False,)

        # initial conditions
        ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)
        
        #pumping
        wel = flopy.mf6.ModflowGwfwel(gwf,
                                          stress_period_data={0: [[(0, pumping_welr, pumping_welc), -0.07, 0]]},
                                      pname='WEL',
                                      auxiliary='CONCENTRATION')

        sto = flopy.mf6.ModflowGwfsto(gwf,
                                      ss=ss_new,
                                      steady_state={0:True})


        # output control
        oc = flopy.mf6.ModflowGwfoc(gwf,
                                    head_filerecord='{}.hds'.format(gwfname),
                                    budget_filerecord='{}.cbc'.format(gwfname),
                                    saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],)

        sim.write_simulation()
        sim.run_simulation()

        mp = flopy.modpath.Modpath7.create_mp7(modelname='modpath', trackdir='forward',
        flowmodel=gwf, model_ws=workspace,
        rowcelldivisions=1, columncelldivisions=1, layercelldivisions=1,
        exe_name="mp7")
        flopy.modpath.mp7bas.Modpath7Bas(mp, porosity=porosity_new, defaultiface=None, extension='mpbas')

        mp.write_input()
        mp.run_model()

    def run_steady(self, img, name):
        mf6exe = 'mf6'
        workspace = self.modflow_workspace(name)
        k = self.get_k(img)
        porosity = self.get_porosity(img)
        ss = self.get_specificstorage(img)
        k_new, porosity_new, ss_new = self.scale(k=k, porosity=porosity, ss=ss, n=self.cell_divisions)
        # spatial discretization
        nlay, nrow, ncol = 1, *np.shape(k_new)
        delr = 500. / nrow
        delc = 500. / ncol
        top = 10.
        botm = 0.

        pumping_welr, pumping_welc = self.pumping_well(n=self.cell_divisions)
        injection_welr, injection_welc = self.injection_well(n=self.cell_divisions)

        
        #initial condition
        strt = 10.
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
        sim = flopy.mf6.MFSimulation(sim_name=name, version='mf6',
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
        gwf = flopy.mf6.MFModel(sim, model_type='gwf6', modelname=gwfname, exe_name='mf6')
        
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
        npf = flopy.mf6.ModflowGwfnpf(gwf, k=k_new, save_specific_discharge=True, save_flows=True,)

        # boundary conditions
        chdlist = [[(0, i, 0), 0.5] for i in range(1, nrow-1)]
        chdlist += [[(0, j, ncol-1), 0.] for j in range(1, nrow-1)]
        chdlist += [[(0, 0 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        chdlist += [[(0, nrow-1 ,i), v] for i, v in zip(range(ncol), np.linspace(0.5, 0, ncol))]
        
        chd = flopy.mf6.ModflowGwfchd(gwf,
                                      stress_period_data=chdlist,
                                      save_flows=False,)

        # initial conditions
        ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)
        
        #pumping
        wel = flopy.mf6.ModflowGwfwel(gwf,
                                          stress_period_data={0: [[(0, pumping_welr, pumping_welc), -0.07]]},
                                          pname='WEL')

        sto = flopy.mf6.ModflowGwfsto(gwf,
                                      ss=ss_new,
                                      steady_state={0:True})


        # output control
        oc = flopy.mf6.ModflowGwfoc(gwf,
                                    head_filerecord='{}.hds'.format(gwfname),
                                    budget_filerecord='{}.cbc'.format(gwfname),
                                    saverecord=[('HEAD', 'ALL')],)

        sim.write_simulation()
        sim.run_simulation()

    #def get_forward_track(self, name, dest_cells=99*200 + 149, days=20.):
    def get_forward_track(self, name, dest_cells=49*100 + 74, days=10.):
        fpth = os.path.join(self.modpath_workspace(name), 'modpath.mpend')
        e = flopy.utils.EndpointFile(fpth)
        well_epd = e.get_destination_endpoint_data(dest_cells=dest_cells)
        return well_epd[well_epd['time']<days*24*3600]
        
    def get_heads(self, name):
        fpth = os.path.join(self.modflow_workspace(name),f'gwf_{name}.hds')
        hdobj = flopy.utils.HeadFile(fpth)
        return hdobj.get_alldata()
    
    def get_times_conc(self, loc_list, name):
        # get the file
        fname = os.path.join(self.modflow_workspace(name), f'gwt_{name}.ucn')
        # extract the data from the file
        concobj = flopy.utils.HeadFile(fname, text='CONCENTRATION')
        # extract the data from the flopy object
        conc = concobj.get_ts(loc_list)
        # make sure concentrations are non-negative
        conc[conc < 0] = 0
        return conc
    
    def get_concentrations(self, name):
        # get the file
        fname = os.path.join(self.modflow_workspace(name), f'gwt_{name}.ucn')
        # extract the data from the file
        concobj = flopy.utils.HeadFile(fname, text='CONCENTRATION')
        # extract the data from the flopy object
        heads = concobj.get_alldata()
        # make sure concentrations are non-negative
        heads[heads < 0] = 0
        return heads
    
    def get_times(self, name):
        fname = os.path.join(self.modflow_workspace(name), f'gwt_{name}.ucn')
        concobj = flopy.utils.HeadFile(fname, text='CONCENTRATION')
        return concobj.get_times()
    
    def get_budget(self, name):
        fpth = os.path.join(self.modflow_workspace(name),f'gwf_{name}.lst')
        print(fpth)
        mflst = flopy.utils.Mf6ListBudget(fpth, timeunit='seconds')
        return mflst.get_dataframes()
    
    def remove_modflow_workspace(self, name):
        shutil.rmtree(self.modflow_workspace(name))
        
    def remove_modpath_workspace(self, name):
        shutil.rmtree(self.modpath_workspace(name))
        
    def log_likelihood(self, name):

        pump_y, pump_x = self.pumping_well()
        # retrieve concentration curve
        times_conc = self.get_times_conc(name=name, loc_list = [(0, pump_y, pump_x)])

        # compute likelihood
        log_likelihood = -np.sum( ((self.measurement[:,0] - times_conc[:,1])**2)) /(2*(0.05e-5)**2)

        # save concentration
        with open(self.concentration_file(name), 'wb') as file_handle:
            pickle.dump(times_conc[:,1], file_handle)

        return log_likelihood
        
    #def get_k(self, val):
        # we flip to plot the same thing as geone
        # modflow arrays have standard layout, if plotted
        # x goes down, y goes right
        # indexing of arrays is [t,z,y,x]
        #val = np.array(val, dtype='float')
        #k = np.flip(10**(val[0,0,:,:]-5),axis=0)
        #k[k==(10**(-4))] = 10**(-5)
        #return k
        
    def predict(self, name, img):
        if self.modpath == 'transient':
            self.run_modpath_transient(name=name, img=img)
            print('Using modpath transient')
        else:
            self.run_modpath(name=name, img=img)
            print('Using modpath steady-state')
            
        try:
            zone = self.get_forward_track(name=name, days=10)
            with open(self.zone_file(name=name), 'wb') as file_handle:
                pickle.dump(zone, file_handle)
        except ValueError as e:
            print(f'{repr(e)} in model {name}', file=sys.stderr)
            

    def compute_log_p_lik(self, model, imod):
        # prepare input
        img = model[0].param_val.reshape(self.ny, self.nx)
        name = f'flow-{imod}'

        # run modflow
        self.run(img=img, name=name)

        # compute likelihood
        log_lik = self.log_likelihood(name=name)
        
        name_mp = f'zone-{imod}'
        # bonus: run modpath
        modpath = TracerTest(cell_divisions=1, modpath=self.modpath, working_dir=self.working_dir)
        modpath.predict(name=name_mp, img=img)

        # clean-up
        self.remove_modflow_workspace(name=name)
        modpath.remove_modpath_workspace(name=name_mp)
        
        return log_lik
    
    def scale(self, k, porosity, ss, n):
        new_porosity = porosity.copy()
        new_porosity = np.repeat(new_porosity, n, axis=0)
        new_porosity = np.repeat(new_porosity, n, axis=1)
        new_k = k.copy()
        new_k = np.repeat(new_k, n, axis=0)
        new_k = np.repeat(new_k, n, axis=1)
        new_ss = ss.copy()
        new_ss = np.repeat(new_ss, n, axis=0)
        new_ss = np.repeat(new_ss, n, axis=1)
        return new_k, new_porosity, new_ss
    
    def pumping_well(self, n=5):
        delr = 5. / n
        delc = 5. / n
        pumping_welr = int(250. / delc) - 1
        pumping_welc = int(375. / delr) - 1
        return pumping_welr, pumping_welc

    def injection_well(self, n=5):
        delr = 5. / n
        delc = 5. / n
        injection_welr = int(250. / delc) - 1        
        injection_welc = int(125. / delr) - 1
        return injection_welr, injection_welc

    def _get_backward_track(self):
        fpth = os.path.join('working/tracertest-mp', 'modpath.mppth')
        p = flopy.utils.PathlineFile(fpth)
        pwb = p.get_destination_pathline_data(dest_cells=249*500+374)

        fpth = os.path.join('working/tracertest-mp', 'modpath.mpend')
        e = flopy.utils.EndpointFile(fpth)
        ewb = e.get_destination_endpoint_data(dest_cells=249*500+374, source=True)

        pwb_new = [p[p['time']<10*24*3600] for p in pwb]

        f, axes = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))

        ax = axes
        ax.set_aspect('equal')
        ax.set_title('Well recharge area')
        mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
        mm.plot_grid(lw=0.5)
        mm.plot_pathline(pwb_new, layer='all', color='blue', lw=0.5, linestyle=':', label='captured by wells')
        #mm.plot_endpoint(ewb, direction='ending') #, colorbar=True, shrink=0.5);

        plt.scatter([125,375], [250,250], marker='x', c='black', s=150)

        plt.tight_layout();
        
        
    def get_k(self, img):
        # we flip to plot the same thing as geone
        # modflow arrays have standard layout, if plotted
        # x goes down, y goes right
        # indexing of arrays is [t,z,y,x]
        k = np.array(img, dtype='float')
        k[k==4] = 1e-1
        k[k==3] = 1e-3
        k[k==2] = 1e-4
        k[k==1] = 1e-5
        return k
    
    def get_porosity(self, img):
        porosity = np.array(img, dtype='float')
        porosity[porosity==4] = 0.25
        porosity[porosity==3] = 0.30
        porosity[porosity==2] = 0.35
        porosity[porosity==1] = 0.40
        return porosity

    def get_specificstorage(self, img):
        ss = np.array(img, dtype='float')
        ss[ss==4] = 1e-5
        ss[ss==3] = 1e-4
        ss[ss==2] = 5e-4
        ss[ss==1] = 1e-3
        return ss
    
    def modflow_workspace(self, name):
        return os.path.join(self.working_dir, 'modflow', name)

    def concentration_file(self, name):
        return os.path.join(self.working_dir, f'{name}-concentration.pickle')
    
    def zone_file(self, name):
        return os.path.join(self.working_dir, f'{name}-zone.pickle')
    
    def modpath_workspace(self, name):
        return os.path.join(self.working_dir, 'modpath', name)
    
    def steady_workspace(self, name):
        return os.path.join(self.working_dir, 'flow', name)
