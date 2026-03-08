import sys
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import os
import ase
from ase.io import read
from ase.data import atomic_numbers, covalent_radii

def structure_process(filename, kind1, kind2, up_down_boundary):
    md_traj = read(str(filename), index=':')
    steps = len(md_traj)
    print(steps)
    middle_layer_total = []
    for i in range(steps):
        temp_traj = md_traj[i]
        middle_layer = process_atoms(kind1, kind2, temp_traj, up_down_boundary)
        middle_layer_total.append(middle_layer)
    middle_layer_total = np.array(middle_layer_total).reshape(steps, -1)

    return middle_layer_total

def process_atoms(kind1, kind2, temp_traj, up_down_boundary):
    # here the kind1 represents the atom type of the Cu layer, kind2 represents the atom type of the S layer
    # temp_traj is the trajectory of the trj[i]
    # up_down_boundary is the boundary of the upper and lower layer of the S layer
    kind1_copy = temp_traj.copy()
    kind2_copy = temp_traj.copy()
    del kind1_copy[[atom.index for atom in kind1_copy if atom.symbol !=kind1]]
    del kind2_copy[[atom.index for atom in kind2_copy if atom.symbol !=kind2]]
    kind1_direct = kind1_copy.get_scaled_positions()
    kind2_direct = kind2_copy.get_scaled_positions()

    up_side = []
    lower_side = []
    for atom in kind2_direct:
        if atom[2] > up_down_boundary:
            up_side.append(atom)
        else:
            lower_side.append(atom)
    
    up_side = np.array(up_side).reshape(-1, 3)
    lower_side = np.array(lower_side).reshape(-1, 3)
    up_average = np.mean(up_side, axis=0)
    lower_average = np.mean(lower_side, axis=0)
    scale = up_average - lower_average
    middle_layer = []
    for atom in kind1_direct:
        middle_layer.append((atom[2]-lower_average[2])/scale[2])
    middle_layer = np.array(middle_layer)

    return middle_layer

def draw_picture(middle_layer_total, output_name):
    steps = len(middle_layer_total)
    x = np.arange(steps)
    y_number = middle_layer_total.shape[1]
    for i in range(y_number):
        y_plot = middle_layer_total[:, i]
        plt.plot(x, y_plot, label='atom'+str(i), linewidth=1.5)
    plt.xlabel('steps (fs)')
    plt.ylabel('Relative position')
    plt.title(output_name)
    plt.savefig(str(output_name))
    plt.show()

def draw_picture_fine(middle_layer_total, output_name):
    steps = len(middle_layer_total)
    x = np.arange(steps)/10000
    y_number = middle_layer_total.shape[1]
    #colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 22,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(10, 10)
    ax = plt.subplot(gs[1:9, 1:9])
    for i in range(y_number):
        y_plot = middle_layer_total[:, i]
        ax.plot(x, y_plot, label='atom'+str(i), linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.25, 1.25)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.set_xlabel('steps (ps)', fontsize=28)
    ax.set_ylabel('Relative position', fontsize=28)
    ax.set_title(output_name, pad=20)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)

    #ax.legend()
    plt.savefig(output_name)
    return

if __name__ == '__main__':
    temp = sys.argv[1]
    start_time = 50000
    end_time = 100000
    filename = 'XDATCAR'
    outfilename = 'carbon_' + str(temp) +'_up.png'
    kind1 = 'Cu'
    kind2 = 'S'
    up_down_boundary = 0.225
    if not os.path.exists("position.npy"):
        print('The file does not exist!')
        # the boundary of the upper and lower layer of the S layer, please modify it according to the real situation
        middle_layer_total = structure_process(filename, kind1, kind2, up_down_boundary)
        np.save('position.npy', middle_layer_total)
    else:
        middle_layer_total = np.load('position.npy')
    part_middle_layer_total = middle_layer_total[start_time:end_time]
    average_middle_layer = np.average(part_middle_layer_total)
    print('temperature', temp)
    print(average_middle_layer)
    draw_picture_fine(part_middle_layer_total, outfilename)
