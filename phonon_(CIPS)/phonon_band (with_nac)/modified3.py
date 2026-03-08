#!/usr/bin/env python

import numpy as np
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib as mpl
#mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.style.use('dark_background')
plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False

import os, yaml
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

############################################################

def read_ph_yaml(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.xz' or ext == '.lzma':
        try:
            import lzma
        except ImportError:
            raise("Reading a lzma compressed file is not supported "
                  "by this python version.")
        with lzma.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    elif ext == '.gz':
        import gzip
        with gzip.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    else:
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=Loader)

    freqs   = []
    dists   = []
    qpoints = []
    labels  = []
    eigvec  = []
    Acell   = np.array(data['lattice'])
    Bcell   = np.array(data['reciprocal_lattice'])

    for j, v in enumerate(data['phonon']):
        if 'label' in v:
            labels.append(v['label'])
        else:
            labels.append(None)
        freqs.append([f['frequency'] for f in v['band']])
        if 'eigenvector' in v['band'][0]:
            eigvec.append([np.array(f['eigenvector']) for f in v['band']])
        qpoints.append(v['q-position'])
        dists.append(v['distance'])

    if all(x is None for x in labels):
        if 'labels' in data:
            ss = np.array(data['labels'])
            labels = list(ss[0])
            for ii, f in enumerate(ss[:-1,1] == ss[1:,0]):
                if not f:
                    labels[-1] += r'|' + ss[ii+1, 0]
                labels.append(ss[ii+1, 1])
        else:
            labels = []

    return (Bcell,
            np.array(dists),
            np.array(freqs),
            np.array(qpoints),
            data['segment_nqpoint'],
            labels, eigvec)


############################################################
# the phonon band
if not os.path.isfile('F1.npy'):
    Bcell, D1, F1_THz, Q1, B1, L1, E1 = read_ph_yaml('../withnac/band.yaml')
    F1 = F1_THz * 33.356
    assert E1, "PHONON EIGENVECTORs MUST NOT BE EMPTY!"
    E1 = np.asarray(E1)
    np.save("Bcell.npy", Bcell) 
    np.save("D1.npy", D1)
    np.save("F1.npy", F1)
    np.save("Q1.npy", Q1)
    np.save("B1.npy", B1) 
    np.save("L1.npy", L1) 
    np.save("E1.npy", E1)

else:
    Bcell = np.load("Bcell.npy")
    D1    = np.load("D1.npy")
    F1    = np.load("F1.npy")
    Q1    = np.load("Q1.npy")
    B1    = np.load("B1.npy")
    L1    = np.load("L1.npy")
    E1    = np.load("E1.npy")

nqpts, nbnds, natoms, _, _ = E1.shape
# real and imaginary  part of the phonon polarization vector
preal, pimag = E1[..., 0], E1[..., 1]

RatioLT = np.zeros_like(F1)
for ii in range(nqpts):
    q = np.dot(Q1[ii], Bcell)
    # exclude Gamma point
    if np.linalg.norm(q) > 1E-10:
        RatioLT[ii,:] = np.linalg.norm(
            (preal[ii] + 1j * pimag[ii]) * q / np.linalg.norm(q),
            axis=(1, 2)
        )
    # exclude Gamma point
    else:
        RatioLT[ii,:] = 0.5


############################################################
fig = plt.figure(
    figsize=(6, 8),
    dpi=480,
    # constrained_layout=True,
)

ax = plt.subplot()

############################################################

for ii in range(0, F1.shape[1]):
    ik = 0
    for nseg in B1:
        ax.plot(
            D1[ik:ik+nseg],
            F1[ik:ik+nseg,ii],
            lw=0.5, color='k', alpha=0.6
        )
        ik += nseg

for ii in np.cumsum(B1)[:-1]:
    ax.axvline(
        x=D1[ii], ls='--',
        color='gray', alpha=0.8, lw=0.5
    )

img = ax.scatter(
    np.tile(D1, (nbnds, 1)).T,
    F1,
    s=np.abs(RatioLT-0.5)*20,
    lw=0,
    c=RatioLT,
    cmap='seismic', alpha=0.6,
    vmin=0, vmax=1.0
)
divider = make_axes_locatable(ax)
ax_cbar = divider.append_axes('right', size='3%', pad=0.02)
cbar = plt.colorbar(img, cax=ax_cbar, ticks=[0, 1])
cbar.set_ticklabels(['T', 'L'])

total_ticks = D1[np.r_[[0], np.cumsum(B1)-1]]
part_ticks = total_ticks[0:4]
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
# ax.set_xlim(D1.min(), D1.max())
ax.set_xlim(D1.min(), total_ticks[3])
ax.set_ylim(-5,375)
ax.set_facecolor('white')
ax.spines[:].set_color('black')
ax.spines[:].set_visible(True)
# ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
ax.set_xticks(part_ticks)

print(part_ticks)
#if L1.all():
#    ax.set_xticklabels(L1)
ax.set_xticklabels(L1[0:4])
print(L1[0:4])
ax.set_ylabel('Frequency (cm$^{-1}$)', labelpad=5)


############################################################
plt.tight_layout(pad=0.5)
plt.savefig('lt_s1.png')
plt.show()
# from subprocess import call
# call('feh -xdF lt_s.png'.split())
