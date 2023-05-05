'''
read eam fs file according to https://sites.google.com/a/ncsu.edu/cjobrien/tutorials-and-guides/eam
'''


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def write(F):
    out = ''
    for i, e in enumerate(F):
        out += u"% 20.16e" % e
        if (i+1)%5 == 0:
            out += '\n'
        else:
            out += '  '
    return out

name = r'C:\Users\user\Desktop\PHTI\GrainGrowth/Ag-Ni.eam.fs.txt'
outname = r'C:\Users\user\Desktop\PHTI\GrainGrowth/Ag-Ni_nonint.eam.fs.txt'
nstart0 = 4

with open(name, 'r') as f:
    src = f.read()
 
lines = src.split('\n')

nstart = nstart0

_t = lines[nstart].split('  ')
nrho = int(_t[0])
nr = int(_t[2])

nstart = nstart+1

head_a = lines[nstart]

nstart = nstart+1

tot = np.zeros(nrho+2*nr)
count = 0
flag = True

for i, line in enumerate(lines[nstart:]):
    for num in line.split('  '):
        if num != '':
            tot[count] = float(num)
            count += 1
        if count == nrho+2*nr:
            flag = False
            break
    if not flag:
        break
    
F_a = tot[:nrho]
rho_aa = tot[nrho:nrho+nr]
rho_ab = tot[nrho+nr:nrho+2*nr]

nstart = nstart + i + 1

head_b = lines[nstart]

nstart = nstart+1

tot = np.zeros(nrho+2*nr)
count = 0
flag = True

for i, line in enumerate(lines[nstart:]):
    for num in line.split('  '):
        if num != '':
            tot[count] = float(num)
            count += 1
        if count == nrho+2*nr:
            flag = False
            break
    if not flag:
        break
    
F_b = tot[:nrho]
rho_ba = tot[nrho:nrho+nr]
rho_bb = tot[nrho+nr:nrho+2*nr]

nstart = nstart + i + 1

tot = np.zeros(3*nr)
count = 0
flag = True

for i, line in enumerate(lines[nstart:]):
    for num in line.split('  '):
        if num != '':
            tot[count] = float(num)
            count += 1
        if count == nrho+2*nr:
            flag = False
            break
    if not flag:
        break
    
phi_aa = tot[:nr]
phi_ab = tot[nr:2*nr]
phi_bb = tot[2*nr:3*nr]

'''
write 
rho_bb = rho_ab
phi_bb = phi_ab
'''

out = """Modified potential without interaction
pass
pass
"""
out += lines[nstart0-1] + '\n'
out += lines[nstart0] + '\n'
out += head_a + '\n'
out += write(F_a)
out += write(rho_aa)
out += write(rho_ab)
out += head_b + '\n'
out += write(F_b)
out += write(rho_ba)
out += write(rho_ab)

out += write(phi_aa)
out += write(phi_ab)
out += write(phi_ab)

with open(outname, 'w+') as f:
    f.write(out)



















