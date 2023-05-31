import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from scipy import interpolate
from scipy.special import erfc
Na = 6e23
k = 1.380649e-23*Na*1e-3 #kJK-1/mol
T = 300
eV2kJmol = 96.4915666370759

fgbs = np.array([31.5,30.6,28.6])/100

#datas = ['5g', 'nonint', 'range2', 'large', 'reverse']
datas = ['sample1', 'sample2', 'sample3', 'sample1ni', 'sample2ni', 'sample3ni']
#fgbs = np.array([31.5])/100
#datas = ['sample1', 'sample1ni']
titles = datas#['with s-s interaction', 'without s-s interaction']
#Ni
sigma = 17.16141009
alpha = 0.703617558
mu = -25.9113993
gamma = 9.32376616

Ea = -2.9714*eV2kJmol
Eb = -4.3855*eV2kJmol
Eba =  67.29501539
Eab = 158.44592421687

Emax = mu+40
Emin = mu-40
num = 1000
Egb = np.arange(num+1)/num*(Emax-Emin)+Emin

s = slice(0, None)
c_cluster = False

def S(x):
    ans = np.zeros(x.shape)
    mask = (x>0)*(x<1) 
    x = x[mask]
    ans[mask] = x*np.log(x)+(1-x)*np.log(1-x)
    return ans*k*T


def f(E, mu, sigma, alpha):
    #dE = np.mean(np.diff(E))
    fs = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(E-mu)**2/(2*sigma**2))*(
        erfc(-alpha*(E-mu)/(np.sqrt(2)*sigma)))#*dE
    return fs/fs.sum()

Fs = f(Egb, mu, sigma, alpha)

def Xgb_func(g1, Fs, Es, T, xgb):
    xs = 1/(1+np.exp((Es-g1)/(k*T)))
    return np.sum(xs*Fs)-xgb

def Egb_func(Fs, Es, g1, T):
    xs = 1/(1+np.exp((Es-g1)/(k*T)))
    return np.sum(xs*Es*Fs)

full_U = []
full_Wgb = []
full_g1 = []
full_c = []
full_mu = []

for di, data in enumerate(datas):
    df = pd.read_csv(f'{data}.txt', comment='#')
    fgb = fgbs[di%len(fgbs)]
    s = (df['c']/fgb<=9)
    cs = df['c'].values[s]/100
    if c_cluster:
        c_ind = np.sum(cs<c_cluster)
        
    mus = df['mu'].values[s]*eV2kJmol
    Us = df['E'].values[s]*eV2kJmol
    Us = Us-Us[0]#-6.7#-5.5
    
    g1s = []
    g2s = []
    Wgbs = []
    
    for i, c in enumerate(cs):
        if c>0:
            xgb = c/fgb
            mu_e = mus[i]
            
            
            g1 = fsolve(Xgb_func, [0], (Fs, Egb, T, xgb))
            g1s.append(g1[0])
            #print(Xgb_func(g1, Fs, Egb, mu_e, T, xgb))
            
            g2 = Us[i] - (c*(Eb-Ea+Eba) + fgb*(Egb_func(Fs, Egb, g1, T)))
            g2s.append(g2)
            
            #Wgb = (g2+g1*(1-fgb)*dmu/cosh)/(fgb+(1-fgb)*dmu/cosh)
            Wgb = g2/fgb
            Wgbs.append(Wgb)
        
    Wgbs = np.array(Wgbs).ravel()
    g1s = np.array(g1s)
    g2s = np.array(g2s)
    
    xgb = cs[1:]/fgb
    xc = 1/(1+np.exp(-g1s/k/T))
    xc_c = (xc[1:]+xc[:-1])/2
    
    zgb = np.polyfit(xgb, Wgbs, 2)
    pgb = np.poly1d(zgb)
    dzgb = np.array([2*zgb[0], zgb[1]])
    dpgb = np.poly1d(dzgb)
    
    dWgbs = (Wgbs[1:]-Wgbs[:-1])/(xgb[1:]-xgb[:-1])
    xgb_c = (xgb[1:]+xgb[:-1])/2
    
    dWcs = ((mus[1:-1]+mus[2:])/2 - (g1s[:-1]+g1s[1:])/2 - (g2s[:-1]-g2s[1:])/(xgb[1:]-xgb[:-1]))/(1-fgb)
    dzc = np.polyfit(xc_c, dWcs, 1)
    dpc = np.poly1d(dzc)
    zc = np.array([dzc[0]/2, dzc[1], 0])
    pc = np.poly1d(zc)
    
    #plt.figure(dpi=1000, figsize=(7, 7))
    
    plt.subplot(221)
    plt.title('Total energy change')
    plt.plot(cs*100, Us)
    plt.plot(cs*100, Us, '.')
    plt.ylabel('energy, $kJ/mol$')
    
    plt.subplot(223)
    plt.title('Chemical potential')
    plt.plot(cs*100, mus)
    plt.plot(cs*100, mus, '.')
    plt.xlabel(r'concentration, %')
    plt.ylabel('energy, $kJ/mol$')
    
    
    plt.subplot(222)
    plt.title('Interaction energy')
    #plt.plot(xgb*100, Wgbs)
    #pgb = (lambda x: -87*x*x - 2*x)
    plt.plot(xgb*100, pgb(xgb), '--', label='$ W_{gb} = '+f'{round(zgb[0])} x^2 {round(zgb[1])} x'+'$')
    plt.plot(xgb*100, Wgbs, '.')
    
    if c_cluster:
        x1 = xgb[c_ind]*100
        y1 = pgb(xgb[c_ind])
        a=0.3
        x2 = xgb.max()*100*a + x1*(1-a)
        y2 = Wgbs.max()*a+y1*(1-a)
        plt.annotate('cluster', 
                     xy=(x1, y1),
                     xytext=(x2, y2),
                     arrowprops=dict(facecolor='black', width=1, headwidth=5))
    plt.xlim((0, xgb.max()*100))
    
    plt.legend()
    
    
    plt.subplot(224)
    plt.title('Derivative of int. energy')
    x = np.linspace(0, xgb.max())
    plt.plot(x*100, dpgb(x), '--', label="$W'_{gb}$")
    plt.plot(xgb_c*100, dWgbs, '.')
    
    
    if c_cluster:
        y1 = dpgb(x1/100)
        y2 = dWgbs.max()*a+y1*(1-a)
        plt.annotate('cluster', 
                     xy=(x1, y1),
                     xytext=(x2, y2),
                     arrowprops=dict(facecolor='black', width=1, headwidth=5))
    
    #_xc = np.linspace(xc.min(), xc.max())
    plt.plot(xgb_c*100, dpc(xc_c), '--', label="$W'_c = " + f'{round(dzc[1])} '+'$')
    plt.plot(xgb_c*100, dWcs, '.')
    plt.xlim((0, cs.max()*100/fgb))
    plt.xlabel(r'GB concentration, %')
    plt.legend()
    
    plt.suptitle(titles[di])
    plt.tight_layout()
    
    plt.show()
    
    full_U.append(Us[1:])
    full_Wgb.append(Wgbs)
    full_g1.append(g1s)
    full_c.append(cs[1:])
    full_mu.append(mus[1:])
#%%  
U = []
Wgb = []
g1 = []
mu = []
plots = []
cmin, cmax = [], []
for cs in full_c:
    cmin.append(cs.min())
    cmax.append(cs.max())
c = np.linspace(np.max(cmin), np.min(cmax))
    
#c = np.linspace(np.max(np.min(np.array(full_c), axis=1)), np.min(np.max(np.array(full_c), axis=1)))
for i, data in enumerate(datas):
    U.append(interpolate.interp1d(full_c[i], full_U[i]))
    Wgb.append(interpolate.interp1d(full_c[i], full_Wgb[i]))
    g1.append(interpolate.interp1d(full_c[i], full_g1[i]))
    mu.append(interpolate.interp1d(full_c[i], full_mu[i]))
    if i<len(fgbs):
        p = plt.plot(c*100, Wgb[-1](c), label=data)
        plots.append(p)


for i in range(len(fgbs)):
    fgb = fgbs[i]
    Wgb = (U[i](c) - U[i+len(fgbs)](c))/fgb
    print(f'{datas[i]} {datas[i+len(fgbs)]}')
    
    #plt.plot(c*100, Wgb, color=, linestyle='-.')
    z = np.polyfit(c/fgb, Wgb, 2)
    p = np.poly1d(z)
    print(z)
    plt.plot(c*100, p(c/fgb), linestyle='--', color=plots[i][0].get_color(), 
             label=f'${round(z[0], 2)}x^2 + {round(z[1], 2)}x$')
plt.ylabel('Wgb')

plt.legend()
plt.show()
