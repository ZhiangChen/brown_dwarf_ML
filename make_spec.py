from fm_new import *
from numpy.random import *
import pickle
import time
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pyplot import *

########################
#reading in the data
########################
'''
data=pickle.load( open('data0.pic','rb'))
loc=np.where((data[0] < 2.4) & (data [0] > 0.9))[0]
wlgrid0=data[0][loc]
y_meas0=data[1][loc]
err0=data[2][loc]

wl_R,R0=np.loadtxt('RvWL.txt').T
R=np.interp(wlgrid0, wl_R, R0)

#regn wlgrid
wlgrid=[]
wl=wlgrid0[0]
f=interpolate.interp1d(wl_R, R0)
while wl < wlgrid0.max():
	R=f(wl)
	dwl=wl/R
	wl=wl+dwl
	wlgrid=np.append(wlgrid,wl)

y_meas=np.interp(wlgrid, wlgrid0, y_meas0)
err=np.interp(wlgrid, wlgrid0, err0)

#errorbar(wlgrid0,y_meas0,yerr=err0,xerr=None,fmt='ob')
#errorbar(wlgrid,y_meas,yerr=err,xerr=None,fmt='or')
#pickle.dump([wlgrid, y_meas, err],open('data.pic','wb'))
'''

'''
#FROM AGAVE
#11 GB card
interactive -n 8 -p asinghargpu1 -q wildfire  --gres=gpu:1 

#16 GB card
interactive -n 20 -p cidsegpu1 -q wildfire --time=6:00:00 --gres=gpu:1  

module purge
module load tensorflow/1.8-agave-gpu
module unload python/2.7.14
module load multinest/3.10
module load anaconda2/4.4.0




nvprof --print-gpu-trace python make_spec.py

'''


data=pickle.load( open('data.pic','rb'))
loc=np.where((data[0] < 2.5) & (data [0] > 0.95))[0]
wlgrid=data[0][loc][::3]
y_meas=data[1][loc][::3]
err=data[2][loc][::3]

print '=== Loaded Data ==='

#setting up model parameters############
logg=5.0  #logg
Teff=730  #effective temp
#log(VMR(ppm))H2O    CH4  CO   CO2  NH3   K   Na
gas=np.array([2.3,  2.0,  1,  -5,  0.6,  -1.5,  0.5])
gamma=50.  #TP smoothing gamma
logbeta=-4  #TP smoothing prior gamma distribution width
logKc=-10  #log cloud opacity (m2/kg)
logPc=1.5   #cloud base pressure
Hcloud=3   #cloud profile slope (0 is constant, larger is steeper)
R2D2=1./8.89**2  #(Rj/D(pc))**2--radius to distance dillution factor in jupiter radius per parsec

#using initial guess logg and Teff to interpolate from Marley Grid Model
marley_grid=readsav('Marley_TP_grid.save')
garr=marley_grid.Garr
garr=np.log10(garr*100)
Tarr=marley_grid.Tarr
grid=marley_grid.Array
Pgrid=marley_grid.P0
Tint=np.zeros(Pgrid.shape[0])
for i in range(Pgrid.shape[0]):
        f=interpolate.interp2d(garr, Tarr, np.array(grid[:,:,i]))
        Tlev = f(logg, Teff)
        Tint[i]=Tlev
logPP=np.arange(-3.5,2.5,0.4)+0.4  #the parameterized grid...this must also be defined in call_fx
NT=logPP.shape
PP=10**logPP
Tpar = interp(np.log10(PP),np.log10(Pgrid),Tint)
tol=np.log10(np.median(err**2))
shift=0.0
#final initial state vector
x0=np.concatenate([gas,np.array([logg,R2D2]),np.array([shift,tol,gamma,logbeta,logKc,logPc,Hcloud]),Tpar])
#logH2O, logCH4, logCO, logCO2, logNH3, logH2S,logNa, logg,scale, shift, logf,gam,logbeta
#plotting initial guess spectrum###########
x=x0


y_mod,y_hires,wnocrop=fx(x,wlgrid)


for i in range(10):
 	print '*****************'
	t1=time.time()
	y_mod,y_hires,wnocrop=fx(x,wlgrid)
	t2=time.time()
	print(t2-t1)

#model from 10000-4000 wno takes 1.29s


errorbar(wlgrid,y_meas,yerr=err,xerr=None,fmt='o')
plot(wlgrid,y_mod)
axis([0.7,2.8, -0.2*1.5*np.max(y_meas),1.5*np.max(y_meas)])
axhline(y=-0.1*1.5*np.max(y_meas),color='k',ls='--')
#plot(wlgrid, y_mod-y_meas-0.1*1.5*np.max(y_meas))
#plot(1E4/wnocrop, y_hires,color='black')

print sum((y_meas-y_mod)**2/err**2)/wlgrid.shape[0]


show()
pdb.set_trace()

#if makeing fake data
#pickle.dump([wlgrid, y_meas, err],open('data.pic','wb'))

pdb.set_trace()

