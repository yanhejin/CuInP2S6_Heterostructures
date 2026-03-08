import numpy as np
import matplotlib as mpl
#mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt
from matplotlib import ticker

shift_energy = -4.2296

x_mesh=np.loadtxt('X.grd')      
y_mesh=np.loadtxt('Y.grd')   
v_mesh=np.loadtxt('LDOS.grd')        
y_mesh_shift = y_mesh + shift_energy
print(x_mesh)
print(v_mesh.min())

mpl.rcParams['axes.unicode_minus'] = False
font = {'family': 'Arial', 'size': 28,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
plt.rc('font', **font)
#fig, ax2d = plt.subplots()
fig, ax2d = plt.subplots(figsize=(8,8))
levels = np.linspace(v_mesh.min(), v_mesh.max(), 50)
#cmap = plt.cm.get_cmap("viridis") 
cmap= mpl.colormaps.get_cmap("jet")
cs = ax2d.contourf(x_mesh,y_mesh_shift,v_mesh,levels=levels, cmap=cmap)
ax2d.set_xlim([4, 13])
ax2d.set_ylim([-6.3, -2])
ax2d.xaxis.set_major_locator(plt.MultipleLocator(2.5))
ax2d.yaxis.set_major_locator(plt.MultipleLocator(1))
#ax2d.set_aspect(2)  #1 为保持X,Y轴等比例
ax2d.set_xlabel(r'x ($\mathregular{\AA}$)', labelpad=7, fontsize = 34)
ax2d.set_ylabel(r'Energy (eV)', labelpad=7, fontsize = 34)
ax2d.spines['bottom'].set_linewidth(2)
ax2d.spines['top'].set_linewidth(2)
ax2d.spines['left'].set_linewidth(2)
ax2d.spines['right'].set_linewidth(2)
ax2d.tick_params(length = 5, width = 2) # direction = 'in' colors = 'red' 

cbar = fig.colorbar(cs,fraction = 0.025, pad = 0.07)                                     #cbar3
tick_locator = ticker.MaxNLocator(nbins=4)  # colorbar上的刻度值个数  
cbar.outline.set_visible(False)
#cbar.outline.set_linewidth(2)
cbar.locator = tick_locator
#cbar.set_ticks([0,8,16,24])
cbar.ax.tick_params(width = 2,length = 5)
cbar.update_ticks()

#plt.axis('equal')
#plt.axis('off')
#plt.show()
plt.tight_layout(pad=1)
plt.savefig("c1_density2.jpg",dpi=300)
