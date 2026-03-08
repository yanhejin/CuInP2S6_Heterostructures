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

def single_generate(middle_layer_total):
    # middle_layer_total is the total trajectory of the middle layer
    # start_time and end_time are the time range you want to select
    time = middle_layer_total.shape[0]
    single_num = middle_layer_total.shape[1]
    time = end_time-start_time + 1
    single_new = np.zeros((time, single_num))
    for i in range(time):
        for j in range(single_num):
            if middle_layer_total[i][j] > 0.5:
                single_new[i][j] = 0 # energy low state is 0
            else:
                single_new[i][j] = 1 # energy high state is 1
    return single_new

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
    # ax.set_ylim(-0.25, 1.25)
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
    # outfilename = 'jump' + str(temp) +'_up.png'
    # kind1 = 'Cu'
    # kind2 = 'S'
    # up_down_boundary = 0.225
    if not os.path.exists("bit.npy"):
        print('The file does not exist!')
        # the boundary of the upper and lower layer of the S layer, please modify it according to the real situation
        middle_layer_total = np.load('position.npy')
        single_new = single_generate(middle_layer_total)
        np.save('bit.npy', single_new)
    else:
        single_new =  np.load('bit.npy')

    part_single_new = single_new[start_time:end_time]

    unique, times = np.unique(part_single_new, return_counts = True)

    if len(unique) == 1:
        print('temperature:', temp)
        print('Only one state found, no state change detected.')
        exit()
    elif len(unique) > 1:
        print('temperature:', temp)
        print('State change detected, two states found.')
        print('unique:', unique)
        print('frequency:', times)
        print('probability:', times[0]/(times[0]+times[1])) # probability of state 0 (UPPER state in manuscript)

    
    ### Additional analysis of intervals between state changes, may be useful for understanding the signal ...
    # sig_num = part_single_new.shape[1]
    # intervals = []
    # for i in range(sig_num):
    #     temp_changes = np.where(np.diff(part_single_new[:,i]) != 0)[0] + 1
    #     temp_intervals = np.diff(temp_changes)
    #     #print(temp_intervals)
    #     if len(temp_intervals) != 0:
    #         for j in range(len(temp_intervals)):
    #             intervals.append(temp_intervals[j])
    # # print('intervals:', intervals)
    # intervals_array = np.array(intervals)
    # print('intervals:', intervals_array)
    # print('average:', np.average(intervals_array))
    # print('std:', np.std(intervals_array))
    # print('number:', len(intervals_array))
