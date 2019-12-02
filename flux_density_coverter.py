"""
README:
This routine is for flux density unit conversion. 
It converts flux density (in W per m^2 per m) of ground-truth data to Janskys.

"""


import pickle
import numpy as np

data_2MASS_J0050 = pickle.load(open('data_2MASS_J0050.pic','rb'))
data_2MASS_J0415 = pickle.load(open('data_2MASS_J0415.pic','rb'))
data_2MASS_J0727 = pickle.load(open('data_2MASS_J0727.pic','rb'))
data_2MASS_J0729 = pickle.load(open('data_2MASS_J0729.pic','rb'))
data_2MASS_J0939 = pickle.load(open('data_2MASS_J0939.pic','rb'))
data_2MASS_J1114 = pickle.load(open('data_2MASS_J1114.pic','rb'))
data_2MASS_J1217 = pickle.load(open('data_2MASS_J1217.pic','rb'))
data_2MASS_J1553 = pickle.load(open('data_2MASS_J1553.pic','rb'))
data_Gl570D = pickle.load(open('data_Gl570D.pic','rb'))
data_HD3651B = pickle.load(open('data_HD3651B.pic','rb'))
data_PSOJ224 = pickle.load(open('data_PSOJ224.pic','rb'))
data_PSO_J043 = pickle.load(open('data_PSO_J043.pic','rb'))
data_ROSS458C = pickle.load(open('data_ROSS458C.pic','rb'))
data_SDSS1504 = pickle.load(open('data_SDSS1504.pic','rb'))
data_UGPSJ0521 = pickle.load(open('data_UGPSJ0521.pic','rb'))
data_UGPS_J072227 = pickle.load(open('data_UGPS_J072227.pic','rb'))
data_ULASJ1416 = pickle.load(open('data_ULASJ1416.pic','rb'))
data_ULASJ2321 = pickle.load(open('data_ULASJ2321.pic','rb'))
data_WISEJ0521 = pickle.load(open('data_WISEJ0521.pic','rb'))
data_SDSS_J162838 = pickle.load(open('data_SDSS_J162838.pic','rb'))

all_objects = np.array([data_2MASS_J0050,data_2MASS_J0415, data_2MASS_J0727, data_2MASS_J0729, data_2MASS_J0939, data_2MASS_J1114, data_2MASS_J1217, data_2MASS_J1553, data_Gl570D, data_HD3651B, data_PSOJ224, data_PSO_J043, data_ROSS458C, data_SDSS1504, data_SDSS_J162838, data_UGPSJ0521, data_UGPS_J072227, data_ULASJ1416, data_ULASJ2321, data_WISEJ0521])
names = ['2MASS_J0050','2MASS_J0415','2MASS_J0727','2MASS_J0729','2MASS_J0939','2MASS_J1114','2MASS_J1217','2MASS_J1553','Gl570D','HD3651B','PSOJ224','PSO_J043','ROSS458C','SDSS1504','SDSS_J162838','UGPSJ0521','UGPS_J072227','ULASJ1416','ULASJ2321','WISEJ0521']

for i in all_objects:
	for j in names:
		wl_data, flx_data, flxerr_data = i[0], i[1], i[2]	
		c = 3.E8
		flx_data = ((wl_data**2)*(flx_data))/(c*10.**-26) # in janskys (Jy), the err too
		flxerr_data = ((wl_data**2)*(flxerr_data))/(c*10.**-26)
		np.save(j,[wl_data,flx_data,flxerr_data])
