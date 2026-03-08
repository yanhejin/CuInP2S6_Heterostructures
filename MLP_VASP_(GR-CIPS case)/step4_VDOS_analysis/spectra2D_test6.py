import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap

hetero = False
inputfile1 = "pdos.csv"

# hetero=True
# inputfile1="spec1.csv"
# inputfile2="spec2.csv"

left_large = True
dynamic = True
num_contour = 20

def lineplot(spec, freq):
    fig = plt.figure()
    fig.set_size_inches(8, 5)
    font = {'family': 'Arial', 'size': 16,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    ax = plt.subplot()


    # 创建从蓝到红的颜色映射
    colors = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'], N=spec.shape[0])

    # 绘制每一行作为一条曲线，颜色根据行数变化
    freq = np.squeeze(freq)
    print(spec.shape)
    print(freq.shape)
    for i in range(spec.shape[0]):
        ax.plot(freq, 1000*spec[i], color=colors(i / spec.shape[0]), alpha=0.7, linewidth=1.5)
        #ax.plot(freq, 1000*spec[i])
    ax.set_xlabel("Frequency (cm$^{-1}$)", fontsize=20)
    ax.set_ylabel("Intensity (arb units)", fontsize=20)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    # ax.set_aspect('equal')
    ax.set_xlim(25, 375)
    plt.tight_layout()
    plt.savefig('show3.png', dpi=300)

def contourplot(spec,name):
    x = spec.columns[0:].astype(float)
    y = spec.index[0:].astype(float)
    z = 10000*spec.values
    z_fit = gaussian_filter(z, sigma=3, mode='nearest')


    # 创建下三角掩码（不包括对角线）
    mask = np.tri(*z_fit.shape, k=-1).astype(bool)
    #z_fit_masked = np.ma.masked_where(~mask, z_fit)
    z_fit_masked = np.ma.masked_where(mask, z_fit)


    fig = plt.figure()
    fig.set_size_inches(8, 6) # 8, 6
    font = {'family': 'Arial', 'size': 16,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    ax = plt.subplot()
    #cmap = ax.pcolormesh(x, y, z_fit_masked, cmap='bwr', vmin=z_fit.min(), vmax=-z_fit.min())
    cmap = ax.pcolormesh(x, y, z_fit_masked, cmap='bwr', vmin=-0.03, vmax=0.03)
    ax.set_xlabel("Frequency (cm$^{-1}$)", fontsize=20)
    ax.set_ylabel("Frequency (cm$^{-1}$)", fontsize=20)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(25)) 
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))
    ax.set_aspect('equal')
    cb = fig.colorbar(cmap, ax=ax)
    # cb.ax.set_ylim(z_fit.min, z_fit.max)
    cb.set_label('Intensity (arb units)',fontsize=20)
    cb.locator = plt.MaxNLocator(nbins=7)
    cb.update_ticks()
    ax.set_xlim(50, 375)
    ax.set_ylim(50, 375)
    plt.tick_params(pad = 5,width = 2)
    #plt.title("Asynchronous correlation", fontsize=20)
    plt.tight_layout()
    plt.savefig(name, dpi=300)

# file read
spec1 = pandas.read_csv(inputfile1, header=0, index_col=0).T
if hetero == False: inputfile2 = inputfile1
spec2 = pandas.read_csv(inputfile2, header=0, index_col=0).T
if len(spec1) != len(spec2): raise Exception("data mismatching")
# spec1.T.plot(legend=None)
# plt.savefig('show1.png', dpi=300)
spec1_array = spec1.values
freq = spec1.columns.astype(float)
print(spec1_array)
print(freq)
lineplot(spec1_array,freq)
if left_large: plt.xlim(max(spec1.columns), min(spec1.columns))
if hetero:
    spec2.T.plot(legend=None)
    if left_large:
        plt.xlim(max(spec2.columns), min(spec2.columns))
        plt.savefig('show2.png', dpi=300)
if dynamic:
    spec1 = spec1 - spec1.mean()
    spec2 = spec2 - spec2.mean()

# synchronous correlation
sync = pandas.DataFrame(spec1.values.T @ spec2.values / (len(spec1) - 1))
sync.index = spec1.columns
sync.columns = spec2.columns
sync = sync.T
name1 = "pic1_sync.png"
contourplot(sync,name1)
sync.to_csv(inputfile1[: len(inputfile1) - 4] + "_sync.csv")

# Hilbert-Noda transformation matrix
noda = np.zeros((len(spec1), len(spec1)))
for i in range(len(spec1)):
    for j in range(len(spec1)):
        if i != j: noda[i, j] = 1 / math.pi / (j - i)

# asynchronouse correlation
asyn = pandas.DataFrame(spec1.values.T @ noda @ spec2.values / (len(spec1) - 1))
asyn.index = spec1.columns
asyn.columns = spec2.columns
asyn =  asyn.T
name2 = "pic2_asyn.png"
contourplot(asyn,name2)
asyn.to_csv(inputfile1[: len(inputfile1) - 4] + "_async.csv")
