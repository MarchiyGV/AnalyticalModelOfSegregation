import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
from numba import njit
from scipy.optimize import fsolve
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpmath as mp 
Na = 6e23
k = 1.380649e-23*Na*1e-3 #kJK-1/mol
eV2kJmole = 96.4915666370759

#Ni
sigma = 17.16141009
alpha = 0.703617558
mu = -25.9113993
a = 1
# Wgb1 = -5*a#-63.3
# Wgb2 = -71*a
# dWc0 = 47*a

# Wgb1 = -6*a
# Wgb2 = -52*a
# dWc0 = 70*a

# Wgb1 = -3*a#-63.3
# Wgb2 = -96*a
# dWc0 = 60*a#73*a

Wgb1 = -2.14*a#-63.3
Wgb2 = -87.16*a
dWc0 = 72*a#73*a
    
gamma = 9.32376616

Ea = -2.9714*eV2kJmole
Eb = -4.3855*eV2kJmole

mu_new00 = np.array([-10])


Ts = np.linspace(200, 1200, 100)
Xs = np.linspace(0.01, 100, 100)/100


Emax = mu+50
Emin = mu-50
num = 1000
Egb = np.arange(num+1)/num*(Emax-Emin)+Emin
Ec = 0	


Eba_md =  67.29501539
Eab_md = 158.44592421687

L0 = 54.62 + 3e-3*300
L1 = 2.8

eps=1e-15
min_eps = 1e-14
niter = 1e10
iteration_prefactor = 1.5

def f(E, mu, sigma, alpha):
    #dE = np.mean(np.diff(E))
    fs = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(E-mu)**2/(2*sigma**2))*(
        erfc(-alpha*(E-mu)/(np.sqrt(2)*sigma)))#*dE
    return fs/fs.sum()


@njit(cache=True)
def Fgb(x, alpha):
    res = Wgb1*x + x*x*Wgb2
    return res*alpha

@njit(cache=True)
def dFgb(x, alpha):
    res = Wgb1 + 2*x*Wgb2
    
    return res*alpha

@njit(cache=True)
def Fc(x, alpha):
    return alpha*dWc0*x

@njit(cache=True)
def dFc(x, alpha):
    return alpha*dWc0 

@njit(cache=True)
def xgb_func(mu, Fs, Egb, T, x):
    xs = 1/(1+np.exp((Egb-mu)/k/T))
    xgb = np.sum(Fs*xs)
    return xgb - x

@njit(cache=True)
def func(mu_new, Fs, Egb, T, alpha, X):

    xs = 1/(1+np.exp((Egb-mu_new)/k/T))
    xgb = np.sum(Fs*xs)
    xc = 1/(1+np.exp(-mu_new/k/T))
    
    fgb = (X - xc)/(xgb - xc)
    
    Fgb_v = Fgb(xgb, alpha)
    Fc_v = Fc(xc, alpha)
    
    dFgb_v = dFgb(xgb, alpha)
    dFc_v = dFc(xc, alpha)
    dF_v = dFgb_v - dFc_v
    
    mu = mu_new + fgb*dFgb_v + (1-fgb)*dFc_v

    mask0 = (xs==0)
    mask1 = (xs==1)
    mask = (xs>0)*(xs<1)
    g = np.empty(xs.shape)
    g[mask] = Egb[mask]*xs[mask] + k*T*(xs[mask]*np.log(xs[mask])+(1-xs[mask])*np.log(1-xs[mask]))
    g[mask0] = 0
    g[mask1] = Egb[mask1]
    Ggb = np.sum(g*Fs) + Fgb_v
    if xc!=0:
        if xc!=1:
            Gc = Fc_v  + k*T*(xc*np.log(xc) + (1-xc)*np.log(1-xc))
        else:
            Gc = Fc_v
    else:
        Gc = Fc_v
        
    dxc_dxgb = np.sum(Fs*np.cosh(mu_new/k/T)/np.cosh((Egb-mu_new)/k/T))
    return Ggb + gamma - Gc + (mu - fgb*dF_v*(1 - 1/(fgb + (1-fgb)*dxc_dxgb)))*(xc - xgb)


Fs = f(Egb, mu, sigma, alpha)

mu_e = np.empty((len(Ts), len(Xs)))
xgb = np.empty((len(Ts), len(Xs)))
fgb = np.empty((len(Ts), len(Xs)))
xc = np.empty((len(Ts), len(Xs)))
G = np.empty((len(Ts), len(Xs)))
#plt.figure(dpi=1000)

if a == 0:
    for Ti, T in enumerate(Ts):
        _mu_e, _, ier, mesg = fsolve(func, mu_new00, args=(Fs, Egb, T, alpha, 0), 
                                   full_output=True)
        
        for Xi, X in enumerate(Xs):
            mu_e[Ti, Xi] = _mu_e
            
            xs = 1/(1+np.exp((Egb-_mu_e)/k/T))
            xgb[Ti, Xi] = np.sum(Fs*xs)
            xc[Ti, Xi] = 1/(1+np.exp(-_mu_e/k/T))
            dFgb_v = dFgb(xgb[Ti, Xi], 1)
            
            fgb[Ti, Xi] = (X - xc[Ti, Xi])/(xgb[Ti, Xi] - xc[Ti, Xi])
            if fgb[Ti, Xi] < 1:
                g = np.empty(xs.shape)
                mask = (xs==1)
                g[~mask] = Egb[~mask]*xs[~mask] + k*T*(xs[~mask]*np.log(xs[~mask])+(1-xs[~mask])*np.log(1-xs[~mask]))
                g[mask] = (Egb[mask]*xs[mask] + k*T*xs[mask]*np.log(xs[mask]))
                Ggb = np.sum(g*Fs) + Fgb(xgb[Ti, Xi], 1) + gamma
                Gc = Fc(xc[Ti, Xi], 1)
                if xc[Ti, Xi] != 0:
                    Gc += k*T*xc[Ti, Xi]*np.log(xc[Ti, Xi])
                if xc[Ti, Xi] != 1:
                    Gc += k*T*(1-xc[Ti, Xi])*np.log(1-xc[Ti, Xi])
                        
                G[Ti, Xi] = fgb[Ti, Xi]*Ggb + (1-fgb[Ti, Xi])*Gc 
            else:
                xgb[Ti, Xi] = X
                mu_e[Ti, Xi] = fsolve(xgb_func, mu_new00, args=(Fs, Egb, T, X))
                xs = 1/(1+np.exp((Egb-mu_e[Ti, Xi])/k/T))
                xc[Ti, Xi] = 1/(1+np.exp(-mu_e[Ti, Xi]/k/T))
                dFgb_v = dFgb(xgb[Ti, Xi], 1)
                Ggb = np.sum(g*Fs) + Fgb(xgb[Ti, Xi], 1) + gamma
                G[Ti, Xi] = Ggb
            if np.isnan(G[Ti, Xi]):
                print(T, 'nan in G')

for Xi, X in enumerate(Xs):
    for Ti, T in enumerate(Ts):
        mu_new0 = mu_new00
        #print(f'Start iterations for T {T}K')
        dalpha= 1
        alpha = 0
        _mu_e, _, ier, mesg = fsolve(func, mu_new0, args=(Fs, Egb, T, alpha, X), 
                                   full_output=True)
        if ier != 1:
            print(f'!!!!!!! first iteration for T {T} without interaction failed !!!!!!')
            raise RuntimeError()
        else:
            mu_new0 = _mu_e
        ier = 0
        count = 0
        while ier != 1 and count < niter:
            count += 1
            while alpha < 1:
                alpha0 = alpha
                alpha += dalpha
                
                alpha = min(alpha, 1)
                dalpha = alpha - alpha0
                
                _mu_e, _, ier, mesg = fsolve(func, mu_new0, args=(Fs, Egb, T, alpha, X), 
                                           full_output=True)

                if ier != 1:
                    alpha -= dalpha
                    #old_dalpha = dalpha
                    dalpha /= iteration_prefactor
                    msg_flag = True
                    if abs(dalpha)<1e-16:
                        print(f"""
    T {T}. step dalpha on iter {count} is too small! alpha {alpha}
    mu = {_mu_e}
            """)
                        msg_flag = False
                        count = niter
                        break
                    #print(f'# {count}. Reduce dalpha to: {dalpha} at value {alpha}')
                    break
                else:
                    if dalpha*iteration_prefactor+alpha<1:
                        dalpha *= iteration_prefactor
                        count -= 1
                        #print(f'# {count}. Increase dwgb to: {dwgb} at value {wgb}')
                    mu_new0 = _mu_e
                if ier == 0:
                    break
                
        if not ier == 0:
            if count < niter:
                pass
                #print(f'Wgb iteration finished after {count} steps')
            elif msg_flag:
                print(f"""
    T {T}. Number of iterations {niter} exceded! alpha {alpha} dalpha {dalpha}
    mu = {_mu_e}
    """)
        #print(func(ans_z, Fs, Egb, T, 1, X))
        #print(mesg)
        mu_e[Ti, Xi] = _mu_e
        xs = 1/(1+np.exp((Egb-_mu_e)/k/T))
        xgb[Ti, Xi] = np.sum(Fs*xs)
        xc[Ti, Xi] = 1/(1+np.exp(-_mu_e/k/T))
        dFgb_v = dFgb(xgb[Ti, Xi], 1)
        
        fgb[Ti, Xi] = (X - xc[Ti, Xi])/(xgb[Ti, Xi] - xc[Ti, Xi])
        if fgb[Ti, Xi] < 1:
            g = np.empty(xs.shape)
            mask = (xs==1)
            g[~mask] = Egb[~mask]*xs[~mask] + k*T*(xs[~mask]*np.log(xs[~mask])+(1-xs[~mask])*np.log(1-xs[~mask]))
            g[mask] = (Egb[mask]*xs[mask] + k*T*xs[mask]*np.log(xs[mask]))
            Ggb = np.sum(g*Fs) + Fgb(xgb[Ti, Xi], 1) + gamma
            Gc = Fc(xc[Ti, Xi], 1)
            if xc[Ti, Xi] != 0:
                Gc += k*T*xc[Ti, Xi]*np.log(xc[Ti, Xi])
            if xc[Ti, Xi] != 1:
                Gc += k*T*(1-xc[Ti, Xi])*np.log(1-xc[Ti, Xi])
                    
            G[Ti, Xi] = fgb[Ti, Xi]*Ggb + (1-fgb[Ti, Xi])*Gc 
        else:
            xgb[Ti, Xi] = X
            mu_e[Ti, Xi] = fsolve(xgb_func, mu_new00, args=(Fs, Egb, T, X))
            xs = 1/(1+np.exp((Egb-mu_e[Ti, Xi])/k/T))
            xc[Ti, Xi] = 1/(1+np.exp(-mu_e[Ti, Xi]/k/T))
            dFgb_v = dFgb(xgb[Ti, Xi], 1)
            Ggb = np.sum(g*Fs) + Fgb(xgb[Ti, Xi], 1) + gamma
            G[Ti, Xi] = Ggb
        if np.isnan(G[Ti, Xi]):
            print(T, 'nan in G')
    


indexing = 'xy'
Xgrid, Tgrid = np.meshgrid(Xs, Ts, indexing=indexing)

fgb[fgb>1]=1
fgb[fgb<0]=0

#%%
Eba = L0+L1
Eab = L0-L1

# Eba = Eba_md
# Eab = Eab_md

plt.figure(dpi=800)
plt.subplot(221)
im1 = plt.contourf(Xgrid*100, Tgrid, fgb, vmin=0, vmax=fgb.max(), levels=100)
plt.xlabel('$X, \%$')
plt.ylabel('$T, K$')

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                          norm=mpl.colors.Normalize(vmin=0, vmax=fgb.max()))
cbar.set_label('$f_{gb}$', rotation=270, size=12)
cbar.ax.get_yaxis().labelpad = 20
#cbar.set_clim(0,1) 


plt.subplot(223)
# ggb = Ggb[0, :]
# fgb = score[0, :]
# G = ggb*fgb + Eab*Xs*Xs*(1-Xs)+Eba*Xs*(1-Xs)*(1-Xs)
# #plt.plot(Xs, ggb*fgb, label='mix')
# #plt.plot(Xs, Eab*Xs*Xs*(1-Xs)+Eba*Xs*(1-Xs)*(1-Xs), label='cluster')
# plt.plot(Xs*100, G, label='gb mix')
# plt.xlabel('$X, \%$')
# plt.ylabel('$G, kJ/mole$')
# plt.hlines(0, Xs.min()*100, Xs.max()*100, linestyle='--', color='grey')
# x1, x2 = 0, 25
# #plt.xlim((x1, x2))
# #plt.ylim((-0.1, 0.1))
# plt.legend()

cmap = plt.cm.get_cmap("magma")
z = G +Eab*Xgrid*Xgrid*(1-Xgrid)+Eba*Xgrid*(1-Xgrid)*(1-Xgrid)
zmin = z.min()#-10
zmax = z.max()#10

plt.contourf(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=100, cmap=cmap)
plt.xlabel('$X, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                       norm=mpl.colors.Normalize(vmin=zmin, vmax=zmax))
cbar.set_label('$\Delta G, kJ/mole$', rotation=270, size=10)
cbar.ax.get_yaxis().labelpad = 10
#cbar.set_clim(0,1) 


plt.subplot(222)
#plt.title('mu')
z = mu_e
zmin = z.min()#-10
zmax = z.max()#10
plt.contourf(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=100)
plt.xlabel('$X, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                       norm=mpl.colors.Normalize(vmin=zmin, vmax=zmax))
cbar.set_label('$\mu, kJ/mole$', rotation=270, size=10)
cbar.ax.get_yaxis().labelpad = 10
'''
plt.plot(Ts, mu_e)
plt.xlabel('T, K')
plt.ylabel('$\mu, kJ/mole$')
'''

plt.subplot(224)
z = (xgb)*100
zmin = z.min()#-10
zmax = z.max()#10
plt.contourf(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=10)
plt.xlabel('$X, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                       norm=mpl.colors.Normalize(vmin=zmin, vmax=zmax))
cbar.set_label('$X_{gb}, \%$', rotation=270, size=10)
cbar.ax.get_yaxis().labelpad = 10
'''
plt.plot(Ts-273, xgb)
plt.xlabel('$T, ^{\circ}C$')
plt.ylabel('$(1-X_{gb}), \%$')
'''
plt.tight_layout()
plt.show()
#%%

# plt.figure(dpi=800)
# for i in range(0, len(Ts), 10):
#     ggb = Ggb[i, :]
#     fgb = score[i, :]
#     G = ggb*fgb + Eab*Xs*Xs*(1-Xs)+Eba*Xs*(1-Xs)*(1-Xs)
#     #plt.plot(Xs, ggb*fgb, label='mix')
#     #plt.plot(Xs, Eab*Xs*Xs*(1-Xs)+Eba*Xs*(1-Xs)*(1-Xs), label='cluster')
#     plt.plot(Xs*100, G, label=f'{round(Ts[i]/100)*100}K')
# plt.xlabel('$X, \%$')
# plt.ylabel('$G, kJ/mole$')
# plt.hlines(0, Xs.min()*100, Xs.max()*100, linestyle='--', color='grey')
# x1, x2 = 0, 25
# #plt.xlim((x1, x2))
# #plt.ylim((-0.1, 0.1))
# plt.legend()


plt.figure(dpi=800)
im1 = plt.contourf(Xgrid*100, Tgrid, fgb, vmin=0, vmax=fgb.max(), levels=50)
plt.xlabel('$X, \%$')
plt.ylabel('$T, K$')

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                          norm=mpl.colors.Normalize(vmin=0, vmax=fgb.max()))
cbar.set_label('$f_{gb}$', rotation=270, size=12)
cbar.ax.get_yaxis().labelpad = 20
#cbar.set_clim(0,1) 
plt.show()
#%%
Gf = G[10, :]+ Eab*Xs*Xs*(1-Xs)+Eba*Xs*(1-Xs)*(1-Xs)
plt.plot(Xs, Gf)
plt.hlines(0, Xs.min(), Xs.max(), linestyle='--', color='grey')
plt.ylim((-10, 10))
plt.show()