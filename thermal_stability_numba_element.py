import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits import mplot3d
from numba import njit, prange
from copy import deepcopy as cp
from time import time
from matplotlib.ticker import FuncFormatter
                
fmt = lambda x, pos: '{:.2}'.format(x)
name_b = np.array(['Ni', 'Mg', 'Bi', 'Cu', 'Pb', 'La', 'Au', 'Pd', 'Sc', 'Y'])
hm_b = np.array([57720, 1607, 11976, 25696, 32469, -56831, -15599, -20000, -89913, -54585])
hs_b = np.array([40000, 15192, 25074, 12170, 19622, 25961-86.84*1000, -7889, -16402, -6552-96.0029*1e3, 10456-111.54390727682068*1e3])

                
'''
user     
'''
Xs = np.linspace(1, 50, num=100)/100

name = 'Ni'

T = 600 # Temperature
t = 5e-10 #GB thickness (A -> m)

dmax = 1000
'''
code
'''
select = (name_b==name)
hm = hm_b[select][0] #Enthalpy of mixing (J/mol)
hs = hs_b[select][0] #Enthalpy of segregation (J/mol) (in dilute limit)

d = np.linspace(1, dmax, num=1000)*1e-9 # nm -> m
Xgb = np.linspace(0, 1, num=1000)
Md, MXgb = np.meshgrid(d, Xgb, indexing='ij')


@njit(parallel=True)
def argminG(hm, hs, X, Md, MXgb, T, t):
    z = 12
    nu = 0.5
    k = 1.380649e-23*6.242e18 #JK-1 -> eVK-1
    
    Oag = 0.0103*1e-26/6  #m3/kmol -> m3
    Gag = 0.75*6.242e18 #Jm-2 -> eVm-2
    Ga = Gag
    Oa = Oag
    Gb = Ga   
    Ob = Oa
    
    Hm = hm*6.242e18/6.02e23 #J/mol -> eV
    Hs = hs*6.242e18/6.02e23#J/mol -> eV
    
    Mfgb = 1 - ((Md-t)/Md)**3
    MXc = (X-Mfgb*MXgb)/(1-Mfgb)

    vmin = 1e-5
    dG = np.zeros(MXc.shape)
    dGmax=0.1
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
            
    
    wc = Hm/z
    wgb = 2*(wc - Hs/z - (Ob*Gb - Oa*Ga)/(2*t*z))
    
    MGc = z*wc*MXc*(1-MXc) + k*T*(MXc*np.log(MXc) + (1-MXc)*np.log(1-MXc))
    MGgb = z*wgb*MXgb*(1-MXgb) + k*T*(MXgb*np.log(MXgb) + (1-MXgb)*np.log(1-MXgb))
    MGgb += Oa*Ga*(1-MXgb)/t + Ob*Gb*MXgb/t
    G = (1-Mfgb)*MGc + Mfgb*MGgb
    G += z*nu*Mfgb*(MXgb-MXc)*((2*MXgb-1)*wgb - (Ob*Gb-Oa*Ga)/(z*t))
    G += dG
    
    ind = np.argmin(G)
    I = ind//G.shape[1]
    J = ind%G.shape[1]
    
    x = Md[I][J]*1e9
    y = MXgb[I][J]*100
        
    Mg = Ga-t*(MXgb/Oa)*(Hs + k*T*np.log(MXc))
    #Mg = Ga-t*((MXgb-MXc)/((1-MXc)*Oa))*(Hs + k*T*np.log(MXc))
    g = Mg[I][J]
    return x, y, g, G, Mg, I, J

t0=time()

D0 = np.zeros((Xs.shape[0]))
X0 = np.zeros((Xs.shape[0]))
g = np.zeros((Xs.shape[0]))
Gnc = []
Gss = []
Mg = []

for i in range(Xs.shape[0]):
    ans = argminG(hm, hs, Xs[i], cp(Md), cp(MXgb), T, t)
    D0[i] += ans[0]
    X0[i] = ans[1]
    g[i] = ans[2]
    
    Mg.append(ans[4])
    x = ans[0]
    y = ans[1]
    I = ans[5]
    J = ans[6]
    Gnc.append((ans[3])[I][J])
    Gss.append((ans[3])[-1][J])
    '''            
    vmin = np.nanmin(G[i])
    vmax = vmin+1*(np.nanmax(G[i])-vmin)#vmin + 0.1*(G.max()-G.min())
    levels = 500
    level_boundaries = np.linspace(vmin, vmax, levels + 1)

    plt.contourf(d*1e9, Xgb*100, np.transpose(G[i]), level_boundaries, vmin=vmin, vmax=vmax,extend='max')
    plt.plot(x, y, 'o', color='red')
    plt.xlabel('Grain size, nm')
    plt.ylabel('Solute concentrations at GB, %')
    cbar = plt.colorbar(format=FuncFormatter(fmt))
    cbar.ax.set_ylabel('$\Delta G_{mix}, eV$', rotation=90)
    plt.title(f'Global solute concentration {round(100*Xs[i], 2)}%, T = {T}К')
    plt.show()
    '''
    '''
    plt.plot(d*1e9, G[i][:, J])
    plt.plot(x, G[i][I, J], 'x')
    plt.ylim((G[i][I, J], G[i][I, J]+0.00001))
    plt.show()
    
    plt.plot(Xgb*1e2, G[i][I, :])
    plt.plot(y, G[i][I, J], 'x')
    plt.show()
    '''
    '''
    vmin = np.nanmin(Mg[i])
    vmax = vmin+1*(np.nanmax(Mg[i])-vmin)#vmin + 0.1*(Mg.max()-Mg.min())
    levels = 500
    level_boundaries = np.linspace(vmin, vmax, levels + 1)

    plt.contourf(d*1e9, Xgb*100, np.transpose(Mg[i]), level_boundaries, vmin=vmin, vmax=vmax,extend='max')
    plt.plot(x, y, 'o', color='red')
    plt.xlabel('Grain size, nm')
    plt.ylabel('Solute concentrations at GB, %')
    cbar = plt.colorbar(format=FuncFormatter(fmt))
    cbar.ax.set_ylabel('$\gamma eV/m^2$', rotation=90)
    plt.title(f'Global solute concentration {round(100*Xs[i], 2)}%, T = {T}К')
    plt.show()
    '''
print(time()-t0)
'''
plt.plot(Xs*100, Gnc)
plt.plot(Xs*100, Gss, '--')
'''
points_x = [0, 33.3, 50]
points_y = [0, -0.299, -0.254]
'''
points_x = [0, 20, 33.3, 50]
points_y = [0, -0.2, -0.302, -0.287]
'''
#plt.plot(points_x, points_y)
#plt.ylim((0,1000))
#plt.show()
ymin = min(np.min(Gnc), np.min(Gss), np.min(points_y))

ymax = max(np.max(Gnc), np.max(Gss), 0)
#plt.plot(Xs*100, (D0-D0.min())+ymin)

#plt.show()
plt.plot(Xs*100, D0, 'x')
plt.ylim((0, 10))
plt.show()



    
    
    
    
    