import numpy as np
import geone
from popex.popex_objects import CatParam

class DeesseSimulator():
    def __init__(self, nthreads=1):
        self.nthreads = nthreads
        self.nx = 128
        self.ny = 128
        self.nz = 1
        self.sx = 1.
        self.sy = 1.
        self.sz = 1.
        self.nv = 1
        self.varname = "facies"
        self.TI = geone.img.readImageGslib('../ti/strebelle.gslib')

    def run(self, seed, dataImage=None):
        deesse_input = geone.deesseinterface.DeesseInput(nx=self.nx, ny=self.ny, nz=self.nz,
                                                     nTI=1,
                                                     TI=self.TI,
                                                        searchNeighborhoodParameters=geone.deesseinterface.SearchNeighborhoodParameters(rx=40.,ry=40.),
                                                     maxScanFraction=0.25,
                                                     distanceThreshold=0.02,
                                                     nneighboringNode=20,
                                                     distanceType=0,
                                                     seed=seed,
                                                     dataImage=dataImage,
                                                     nrealization=1,
                                                     nv=self.nv,
                                                     varname=self.varname)
        return geone.deesseinterface.deesseRun(deesse_input, verbose=0, nthreads=self.nthreads)['sim'][0]
    
    def generate_m(self, hd_param_ind=(None,), hd_param_val=(None,), imod=0):
        categories = [[(-0.5, 0.5)], [(0.5, 1.5)]]
        seed = 2000 + imod
        
        indexes = hd_param_ind[0]
        values = hd_param_val[0]
        val = np.empty(self.nx*self.ny)
        val[:] = np.nan
        val[indexes] = values
        dataImage = geone.img.Img(nx=self.nx, ny=self.ny, nz=self.nz, sx=self.sx, sy=self.sy, sz=self.sz, nv=self.nv,
                val=val.reshape(self.nv, self.nz, self.ny, self.nx),
                varname=self.varname,
                name="conditioning")
        
        sim = self.run(seed=seed, dataImage=[dataImage])
        
        return (CatParam(param_val=sim.val.reshape(-1), dtype_val='int8', categories=categories),)
    
    def get_hd_pri(self):
        return (None,), (None,) 
    
