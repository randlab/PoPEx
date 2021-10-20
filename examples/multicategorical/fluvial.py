import geone
import numpy as np
from popex.popex_objects import CatParam

class FluvialSimulation():
    def __init__(self, nthreads=1, test=False):
        self.nthreads = nthreads
        self.nx = 100
        self.ny = 100
        self.nz = 1
        self.sx = 5.
        self.sy = 5.
        self.sz = 1.
        self.nv = 1
        self.varname = "facies"
        self.ti_original = geone.img.readImageGslib('../ti/ti_4cat_wl375_2.gslib')
        self.ti = geone.img.Img(nx=500, ny=800, nz=1, sx=5, sy=5,
                                val=self.ti_original.val[:,:,:,:500], nv=1,
                                varname=['facies'], name='smallfluvial')
        self.pyrGenParams = geone.deesseinterface.PyramidGeneralParameters(
            npyramidLevel=2,                 # number of pyramid levels, additional to the simulation grid
            kx=[2, 2], ky=[2, 2], kz=[0, 0]  # reduction factors from one level to the next one
        )

        self.pyrParams = geone.deesseinterface.PyramidParameters(
            nlevel=2,# number of levels
            pyramidType='categorical_auto', # type of pyramid (accordingly to categorical variable in this example)
            nclass=1,
            classInterval = [np.array([[3.5,4.5]])]
        )

        self.test = test
        
    def get_plain_image(self, img):
        return np.flip(np.array(img.val, dtype='float')[0,0,:,:], axis=0)
        
    def generate_ref(self):
        # this was saved as 'data/fluvial-ref.gslib'
        self.deesse_input.seed = 202013
        self.deesse_input.dataPointSet = None
        return geone.deesseinterface.deesseRun(self.deesse_input, nthreads=self.nthreads)['sim'][0]

    def generate_new_ref(self):
        # this was saved as 'data/fluvial-ref-new.gslib'
        self.deesse_input.seed = 2030
        self.deesse_input.dataPointSet = None
        return geone.deesseinterface.deesseRun(self.deesse_input, nthreads=self.nthreads)['sim'][0]
    
    def generate(self, hd=None, i=0, dataImage=None):
        if self.test is True and i==0:
            seed = 202014 - 101
        else:
            seed = 202014+i

        deesse_input = geone.deesseinterface.DeesseInput(nx=100, ny=100, nz=1,
                                                sx=5., sy=5., sz=1.,
                                                ox=0., oy=0., oz=0.,
                                                 nv=1,
                                                varname='facies',
                                                 nTI=1,
                                                 TI=self.ti_original,
                                                 searchNeighborhoodParameters=geone.deesseinterface.SearchNeighborhoodParameters(radiusMode='manual', rx=40, ry=40, rz=0),
                                                 nneighboringNode=60,
                                                 distanceType=0,
                                                 distanceThreshold=0.01,
                                                 maxScanFraction=0.04,
                                                 npostProcessingPathMax=1,
                                                 pyramidGeneralParameters=self.pyrGenParams,
                                                 pyramidParameters=self.pyrParams,
                                                 dataImage=dataImage,
                                                 dataPointSet=hd,
                                                 seed=seed,
                                                 nrealization=1
                                                             )

        return geone.deesseinterface.deesseRun(deesse_input, nthreads=self.nthreads)['sim'][0]
    
    def generate_with_hd(self, i=0):
        hd=[geone.img.PointSet(npt=2, nv=4, val=np.array([[374.5, 249.5, 0.5, 4], [124.5, 249.5, 0.5, 4]]).transpose(), varname=['X','Y','Z','facies'])]
        return self.generate(hd=hd, i=i)


    def generate_m(self, hd_param_ind, hd_param_val, imod):
        indexes = hd_param_ind[0]
        values = hd_param_val[0]
        val = np.empty(self.nx*self.ny)
        val[:] = np.nan
        val[indexes] = values
        dataImage = geone.img.Img(nx=self.nx, ny=self.ny, nz=self.nz, sx=self.sx, sy=self.sy, sz=self.sz, nv=self.nv,
                val=val.reshape(self.nv, self.nz, self.ny, self.nx),
                varname=self.varname,
                name="conditioning")
        #hd=geone.img.PointSet(npt=2, nv=4, val=np.array([[124.5, 249.5, 0.5, 4], [374.5, 249.5, 0.5, 4]]).transpose(), varname=['X','Y','Z','facies'])

        sim = self.generate(i=imod, dataImage=[dataImage])
        return (CatParam(param_val=sim.val.reshape(-1), dtype_val='int8', categories=[[(0.5, 1.5)], [(1.5, 2.5)], [(2.5, 3.5)], [(3.5, 4.5)]]),)

    def get_hd_pri(self):
        return ([np.ravel_multi_index((0,0,49,24), (1,1,100,100)),np.ravel_multi_index((0,0,49,74),(1,1,100,100))],), ([4,4],) 

