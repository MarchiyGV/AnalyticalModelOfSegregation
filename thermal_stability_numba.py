import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits import mplot3d
from numba import njit, prange
from copy import deepcopy as cp
from time import time
'''
user     
'''
meta = 0
name_b = np.array(['Ni', 'Ni', 'Mg', 'Bi', 'Cu', 'Pb', 'La', 'Au', 'Pd', 'Sc', 'Y'])
hm_b = np.array([57720, 57720, 1607, 11976, 25696, 32469, -56831, -15599, -20000, -89913, -54585])
hs_b = np.array([26000, 16000, 15192, 25074, 12170, 19622, 25961-meta*86.84*1000+meta*45.784*1e3, -7889, -16402, -6552-meta*96.0029*1e3, 10456-meta*111.54390727682068*1e3])


#Xs = np.logspace(-4, -1, num=10)  #Global concentration (fraction)
Xs = np.linspace(0.001, 0.999, num=10)
hms = np.linspace(-100, 75, num=49)*1e3
hss = np.linspace(-30, 50, num=50)*1e3
T = 600 # Temperature
t = 5e-10 #GB thickness (A -> m)

dmax = 1000
hm = 100650 #Enthalpy of mixing (J/mol)
hs = 10361 #Enthalpy of segregation (J/mol) (in dilute limit)
hm = 57720 #Ni
hs = 16142 #Ni

'''
code
'''
d = np.linspace(0.5, dmax, num=100)*1e-9 # nm -> m
Xgb = np.linspace(0, 1, num=100)
Md, MXgb = np.meshgrid(d, Xgb)


#hm = 54620+T*3.1G[i][j]==Gmin
#hs = 16142

@njit(parallel=True)
def argminG(hm, hs, X, Md, MXgb, T, t):
    z = 12
    nu = 0.5
    k = 1.380649e-23*6.242e18 #JK-1 -> eVK-1
    Oag = 0.0103*1e-26/6  #m3/kmol -> m3
    Oni = 0.00665*1e-26/6 #m3/kmol -> m3
    Gag = 0.75*6.242e18 #Jm-2 -> eVm-2
    Gni = 1.1*6.242e18/1.27 #Jm-2 -> eVm-2
    
    Omg = 13.97*1e-26/6
    Gmg = 0.1*6.242e18
    
    Gw = 3.675*6.242e18 
    Ow = 9.53*1e-6/6.02e23
    
    Gti = 1.7*6.242e18 
    Oti = 10.64*1e-6/6e23
    
    Obi = 21.3*1e-6/6e23
    
    Ola = 22.386*1e-6/6e23
    
    Ga = Gag
    Oa = Oag
    
    Gb = Gmg
    Ob = Obi
    
    
    Gb = 0.5*6.242e18
    Ob = 8.25*1e-6/6.02e23
    
    Gb = Ga
    Ob = Oa
    
    Hm = hm*6.242e18/6.02e23 #J/mol -> eV
    Hs = hs*6.242e18/6.02e23#J/mol -> eVdG[i][j]=dGmaxdG[i][j]=dGmax
    #wc = Hm/(z*X*(1-X))
    #wgb = 2*(wc*z - Hs - (Ob*Gb - Oa*Ga)/(2*t))/z
    
    Mfgb = 1 - ((Md-t)/Md)**3
    MXc = (X-Mfgb*MXgb)/(1-Mfgb)

    vmin = 1e-10
    dG = np.zeros(MXc.shape)
    dGmax=10000
    for i in prange(MXc.shape[0]):
        for j in prange(MXc.shape[1]):
            if MXc[i][j]<vmin:
                MXc[i][j]=vmin
                MXgb[i][j]=X/Mfgb[i][j]
                dG[i][j]=dGmax
            elif MXc[i][j]>1-vmin:
                MXc[i][j]=1-vmin
                MXgb[i][j]=(1-X)/(1-Mfgb[i][j])
                dG[i][j]=dGmax
            
    #print(MXc.min(), MXc.max())
    
    wc = Hm/z
    wgb = 2*(wc - Hs/z - (Ob*Gb - Oa*Ga)/(2*t*z))
    
    #print(wc, wgb, wgb/wc)
    
    MGc = z*wc*MXc*(1-MXc) + k*T*(MXc*np.log(MXc) + (1-MXc)*np.log(1-MXc))
    MGgb = z*wgb*MXgb*(1-MXgb) + k*T*(MXgb*np.log(MXgb) + (1-MXgb)*np.log(1-MXgb))
    MGgb += Oa*Ga*(1-MXgb)/t + Ob*Gb*MXgb/t
    G = (1-Mfgb)*MGc + Mfgb*MGgb
    G += z*nu*Mfgb*(MXgb-MXc)*((2*MXgb-1)*wgb - (Ob*Gb-Oa*Ga)/(z*t))
    G += dG
    
    ind = np.argmin(G)
    I = ind//G.shape[0]
    J = ind%G.shape[0]
    
    x = Md[I][J]*1e9
    y = MXgb[I][J]*100
        
    g = Ga-t*(MXgb[I][J]/Oa)*(Hs + k*T*np.log(MXc[I][J]))
    

    return x, y, g, G

t0=time()

D0 = np.zeros((hss.shape[0], hms.shape[0], Xs.shape[0]))
X0 = np.zeros((hss.shape[0], hms.shape[0], Xs.shape[0]))
g = np.zeros((hss.shape[0], hms.shape[0], Xs.shape[0]))
G = []

for i in range(hss.shape[0]):
    G.append([])
    for j in range(hms.shape[0]):
        G[-1].append([])
        for k in range(Xs.shape[0]):
            ans = argminG(hms[j], hss[i], Xs[k], cp(Md), cp(MXgb), T, t)
            D0[i,j,k] += ans[0]
            X0[i,j,k] = ans[1]
            g[i,j,k] = ans[2]
            G[-1][-1].append(ans[3])
            
print(time()-t0)
    


vmin = D0.min()
vmax = dmax
levels = 500
plt.figure(dpi=1000)
level_boundaries = np.linspace(vmin, vmax, levels + 1)
plt.contourf(hms/1e3, -hss/1e3, D0[:,:,:].min(axis=2), level_boundaries, 
             locator=ticker.LogLocator(), vmin=vmin, vmax=vmax, extend='max')
plt.plot(hm_b/1e3, -hs_b/1e3, 'o')
dd = 1
for i, name in enumerate(name_b):
    plt.text(hm_b[i]/1e3+dd, -hs_b[i]/1e3+dd, name)
        
plt.ylabel('$\Delta H_{seg}, kJ/mol$')
plt.xlabel('$\Delta H_{mix}, kJ/mol$')
plt.savefig('map.png')
#plt.show()

'''
ind = np.argmin(D0, axis=2)
Gmin = np.zeros(ind.shape, dtype=np.float128)
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        k = ind[i][j]
        Gmin[i][j] = G[i][j][k]
    
plt.contourf(hms/1e3, hss/1e3, Gmin)
plt.colorbar()
plt.show()
'''
    
    
    
    
    