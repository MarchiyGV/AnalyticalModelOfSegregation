import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
from numba import njit
from scipy.optimize import fsolve
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
Na = 6e23
k = 1.380649e-23*Na*1e-3 #kJK-1/mol
eV2kJmole = 96.4915666370759

xtol = 1e-8 #tolerance of solving equation for mu

plot_zero = True

#N in Ag
sigma = 17.16141009
alpha = 0.703617558
mu = -25.9113993 # mean-energy
gamma = 9.32376616 # k*\gamma - per-atom grain boundary energy
Eba_md =  67.29501539 # E B in A
a = 1# 1 if exists solute-solute interaction, 0 otherwise


#Ag in Ni
# sigma = 56.12596918
# alpha = -2.35401727
# mu = -13.0040431
# gamma = 17.6527277
# Eba_md =  158.4459242
# a = 0


Eba = Eba_md
#interaction parameters: W_{xi} = W_{xi}_1*X_{xi} + W_{xi}_2*X_{xi}^2
Wgb1 = -1.7*a
Wgb2 = -98.3*a

dWc1 = 0
dWc2 = 0
    

mu_new00 = np.array([-30]) #initial guess for mu


Ts = np.linspace(200, 1200, 20) # temperature range
Xs = np.linspace(0.0001, 20, 100)/100 # concentration range


Emax = mu+50
Emin = mu-50
num = 1000
Egb = np.arange(num+1)/num*(Emax-Emin)+Emin
Ec = 0	


eps=1e-15
min_eps = 1e-14
niter = 1e10
iteration_prefactor = 1.5

def f(E, mu, sigma, alpha):
    fs = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(E-mu)**2/(2*sigma**2))*(
        erfc(-alpha*(E-mu)/(np.sqrt(2)*sigma)))
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
    return alpha*(dWc1*x + x**2*dWc2/2)

@njit(cache=True)
def dFc(x, alpha):
    return alpha*(dWc1 + x*dWc2)

@njit(cache=True)
def xgb_func(mu, Fs, Egb, T, x):
    xs = 1/(1+np.exp((Egb-mu)/k/T))
    xgb = np.sum(Fs*xs)
    return xgb - x

@njit(cache=True)
def func(mu, Fs, Egb, T, alpha, X):

    xs = 1/(1+np.exp((Egb-mu)/k/T))
    xgb = np.sum(Fs*xs)
    xc = 1/(1+np.exp(-mu/k/T))
    
    fgb = (X - xc)/(xgb - xc)
    
    Fgb_v = Fgb(xgb, alpha)
    Fc_v = Fc(xc, alpha)
    
    dFgb_v = dFgb(xgb, alpha)
    dFc_v = dFc(xc, alpha)
    
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
        
    dxgb_dxc = np.sum(Fs*(1+np.cosh(-mu/k/T))/(1+np.cosh((Egb-mu)/k/T)))
    return Ggb + gamma - Gc + (mu + (fgb*dFgb_v*dxgb_dxc + (1-fgb)*dFc_v)/(fgb*dxgb_dxc + (1-fgb)))*(xc - xgb)


Fs = f(Egb, mu, sigma, alpha)

mu_e = np.empty((len(Ts), len(Xs)))
xgb = np.empty((len(Ts), len(Xs)))
fgb = np.empty((len(Ts), len(Xs)))
xc = np.empty((len(Ts), len(Xs)))
G = np.empty((len(Ts), len(Xs)))
err = np.empty((len(Ts), len(Xs)))
#plt.figure(dpi=1000)

if a == 0:
    for Ti, T in enumerate(Ts):
        _mu_e, _, ier, mesg = fsolve(func, mu_new00, args=(Fs, Egb, T, alpha, 0), 
                                   full_output=True, xtol=xtol)
        
        for Xi, X in enumerate(Xs):
            mu_e[Ti, Xi] = _mu_e
            
            xs = 1/(1+np.exp((Egb-_mu_e)/k/T))
            xgb[Ti, Xi] = np.sum(Fs*xs)
            xc[Ti, Xi] = 1/(1+np.exp(-_mu_e/k/T))
            
            fgb[Ti, Xi] = (X - xc[Ti, Xi])/(xgb[Ti, Xi] - xc[Ti, Xi])
            if fgb[Ti, Xi] < 1 and fgb[Ti, Xi] > 0:
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
            elif fgb[Ti, Xi]>1:
                xgb[Ti, Xi] = X
                mu_e[Ti, Xi] = fsolve(xgb_func, mu_new00, args=(Fs, Egb, T, X),
                                      xtol=xtol)
                xs = 1/(1+np.exp((Egb-mu_e[Ti, Xi])/k/T))
                xc[Ti, Xi] = 1/(1+np.exp(-mu_e[Ti, Xi]/k/T))
                g = np.empty(xs.shape)
                mask = (xs==1)
                g[~mask] = Egb[~mask]*xs[~mask] + k*T*(xs[~mask]*np.log(xs[~mask])+(1-xs[~mask])*np.log(1-xs[~mask]))
                g[mask] = (Egb[mask]*xs[mask] + k*T*xs[mask]*np.log(xs[mask]))
                Ggb = np.sum(g*Fs) + Fgb(xgb[Ti, Xi], 1) + gamma
                G[Ti, Xi] = Ggb
            elif fgb[Ti, Xi] < 0:
                xc[Ti, Xi] = X
                Gc = Fc(xc[Ti, Xi], 1) + k*T*(xc[Ti, Xi]*np.log(xc[Ti, Xi]) + (1-xc[Ti, Xi])*np.log(1-xc[Ti, Xi]))
                G[Ti, Xi] = Gc
            
            if np.isnan(G[Ti, Xi]):
                print(T, 'nan in G')

for Xi, X in enumerate(Xs):
    for Ti, T in enumerate(Ts):
        fail_flag = False
        mu_new0 = mu_new00
        #print(f'Start iterations for T {T}K')
        dalpha= 1
        alpha = 0
        _mu_e, infodict, ier, mesg = fsolve(func, mu_new0, args=(Fs, Egb, T, alpha, X), 
                                   full_output=True, xtol=xtol)
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
                
                _mu_e, infodict, ier, mesg = fsolve(func, mu_new0, args=(Fs, Egb, T, alpha, X), 
                                           full_output=True, xtol=xtol)

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
                        fail_flag = True
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
        
        fgb[Ti, Xi] = (X - xc[Ti, Xi])/(xgb[Ti, Xi] - xc[Ti, Xi])
        if fgb[Ti, Xi] < 1 and fgb[Ti, Xi] > 0:
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
        elif fgb[Ti, Xi]>1:
            xgb[Ti, Xi] = X
            mu_e[Ti, Xi] = fsolve(xgb_func, mu_new00, args=(Fs, Egb, T, X),
                                  xtol=xtol)
            xs = 1/(1+np.exp((Egb-mu_e[Ti, Xi])/k/T))
            xc[Ti, Xi] = 1/(1+np.exp(-mu_e[Ti, Xi]/k/T))
            g = np.empty(xs.shape)
            mask = (xs==1)
            g[~mask] = Egb[~mask]*xs[~mask] + k*T*(xs[~mask]*np.log(xs[~mask])+(1-xs[~mask])*np.log(1-xs[~mask]))
            g[mask] = (Egb[mask]*xs[mask] + k*T*xs[mask]*np.log(xs[mask]))
            Ggb = np.sum(g*Fs) + Fgb(xgb[Ti, Xi], 1) + gamma
            G[Ti, Xi] = Ggb
        elif fgb[Ti, Xi] < 0:
            xc[Ti, Xi] = X
            mu_e[Ti, Xi] = k*T*np.log(X/(1-X)) 
            xs = 1/(1+np.exp((Egb-_mu_e)/k/T))
            xgb[Ti, Xi] = np.sum(Fs*xs)
            Gc = Fc(xc[Ti, Xi], 1) + k*T*(xc[Ti, Xi]*np.log(xc[Ti, Xi]) + (1-xc[Ti, Xi])*np.log(1-xc[Ti, Xi]))
            G[Ti, Xi] = Gc
            
        if np.isnan(G[Ti, Xi]):
            print(T, 'nan in G')
            
        err[Ti, Xi] = infodict['fvec']

    


indexing = 'xy'
Xgrid, Tgrid = np.meshgrid(Xs, Ts, indexing=indexing)

fgb[fgb>1]=1
fgb[fgb<0]=0

#%%

lvl1 = 15
lvl2 = 50


plt.figure(dpi=1000)
plt.subplot(221)
z = fgb


im1 = plt.contourf(Xgrid*100, Tgrid, z, vmin=0, vmax=fgb.max(), levels=lvl1)
plt.gca().get_xaxis().set_visible(False)
plt.xlabel('$X, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                          norm=mpl.colors.Normalize(vmin=0, vmax=z.max()))
cbar.set_label('$f_{gb}$', rotation=270, size=12)
cbar.ax.get_yaxis().labelpad = 20
#cbar.set_clim(0,1) 


plt.subplot(223)
cmap = plt.cm.get_cmap("magma")
contmap = plt.cm.get_cmap("rainbow")
z = G +Eba*Xgrid

zmin = z.min()
zmax = z.max()



plt.contourf(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=lvl1, cmap=cmap)
CS = plt.contour(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=10, 
                 colors=['blue'], linewidths=2)
levels = CS.levels
if levels.min()<0 and levels.max()>0:
    zero = np.min(np.abs(levels))
    print(f'there is zero contour: {zero}')
else:
    print('zere is not zero contour of Gf')
    zero = 0 

for i, level in enumerate(CS.collections):
    val = CS.levels[i]
    if val != zero or not plot_zero:
        for kp,path in reversed(list(enumerate(level.get_paths()))):
            del(level.get_paths()[kp])

plt.xlabel('$X^{tot}, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                       norm=mpl.colors.Normalize(vmin=zmin, vmax=zmax))
cbar.set_label('$\Delta G_{nc}, kJ/mol$', rotation=270, size=10)
cbar.ax.get_yaxis().labelpad = 10

plt.subplot(222)
z = xc*100
zmin = z.min()
zmax = z.max()
plt.contourf(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=lvl2)
plt.gca().get_yaxis().set_visible(False)
plt.gca().get_xaxis().set_visible(False)
plt.xlabel('$X^{tot}, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                       norm=mpl.colors.Normalize(vmin=zmin, vmax=zmax))
cbar.set_label('$X_{c}, \%$', rotation=270, size=10)
cbar.ax.get_yaxis().labelpad = 10

plt.subplot(224)
z = xgb*100

zmin = z.min()
zmax = z.max()
plt.contourf(Xgrid*100, Tgrid, z, vmin=zmin, vmax=zmax, levels=lvl2)
plt.gca().get_yaxis().set_visible(False)
plt.xlabel('$X^{tot}, \%$')
plt.ylabel('$T, K$')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = mpl.colorbar.ColorbarBase(cax,
                       norm=mpl.colors.Normalize(vmin=zmin, vmax=zmax))
cbar.set_label('$X_{gb}, \%$', rotation=270, size=10)
cbar.ax.get_yaxis().labelpad = 10
plt.tight_layout()
plt.show()

#%%
# out = []
# outT = []
# plt.figure(dpi=800)
# for i in range(0, len(Ts), 4):
#     mask = (fgb[i, :]<1)
#     i0 = np.sum(mask)
#     Gf = G[i, :]+ Eba*Xs#*Xs*(1-Xs)+Eba*Xs*(1-Xs)*(1-Xs)
#     G0 = Gf*(1-Xs)
#     plt.plot(Xs*100, Gf, label=f'${round(Ts[i]/100)*100}$'+' K')
#     out.append(Gf)
#     outT.append(Ts[i])
#     #plt.plot(Xs*100, G0, label=f'${round(Ts[i]/100)*100}$'+' K')
#     #plt.plot([Xs[i0]*100], [Gf[i0]], 'o', color='black')
    
# plt.hlines(0, 100*Xs.min(), 100*Xs.max(), linestyle='--', color='grey')
# # plt.ylim((-0.1, 0.1))
# # plt.xlim((0, 1))
#     # plt.twinx()
#     # plt.plot(Xs, fgb[i, :], color='grey')
# plt.xlabel('$X^{tot}, \%$')
# plt.ylabel('$\Delta G_{mix},$'+' kJ/mol')
# plt.legend()
# plt.show()
# np.savetxt('AgNi.csv', out)
# np.savetxt('T.csv', outT)