#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from ase.io import vasp
from ase.constraints import FixScaled

# load the starting geometry
geo0 = vasp.read_vasp('yx_convert_for_shift_CONTCAR')
pc0  = geo0.get_scaled_positions().copy()

# the indeices of the atoms in the surface/bulk layer
# fix the bulk, only release the surface
#Lb   = [ii for ii in range(len(geo0)) if pc0[ii,-1] < 0.2]
#Ls   = [ii for ii in range(len(geo0)) if pc0[ii,-1] >= 0.2]
# only allow movement in z-direction,but also fix the bottom
#cc = []
#for iib in Lb:
#    cc.append(FixScaled(geo0.cell, int(iib), [1, 1, 1]))
#for iis in Ls:
#    cc.append(FixScaled(geo0.cell, int(iis), [1, 1, 0]))
#geo0.set_constraint(cc)

# number of points in x/y direction
nx = ny = 21

# Due to PBC, remove the other borders [:,:-1, :-1]
dxy = np.mgrid[0:1:1j*nx, 0:1:1j*ny][:,:-1,:-1].reshape((2,-1)).T
nxy = np.mgrid[0:nx, 0:ny][:,:-1,:-1].reshape((2,-1)).T
	
L1   = [ii for ii in range(len(geo0)) if pc0[ii,-1] < 0.30]
L2   = [ii for ii in range(len(geo0)) if pc0[ii,-1] >= 0.30]
assert len(L1) + len(L2) == len(geo0)

for ii in range(nxy.shape[0]):
    dx, dy = dxy[ii]
    ix, iy = nxy[ii]

    pc = pc0.copy()
    # only move the atoms in the upper layer
    pc[L2] += [dx, dy, 0]
    geo0.set_scaled_positions(pc)

    # python 3
    os.makedirs('{:02d}-{:02d}'.format(ix, iy), exist_ok=True)
    # python 2
    #if not os.path.isdir('{:02d}-{:02d}'.format(ix, iy)):
    #    os.makedirs('{:02d}-{:02d}'.format(ix, iy))

    vasp.write_vasp("{:02d}-{:02d}/POSCAR".format(ix, iy), geo0, direct=True, vasp5=True)
