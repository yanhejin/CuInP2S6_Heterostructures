import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import os
import ase
from ase.io import read
from ase.data import atomic_numbers, covalent_radii
# from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


############################################
#  Barrier part
############################################

# calculate the free energy barrier
def height_value(unique_elements, relative_gaussian, max_range, min_range):
    # print('Unique elements min:', unique_elements.min())
    if unique_elements.min() < (min_range + 0.3* (max_range - min_range)): # means there has two positions
        print('Two positions')
        # first order derivative
        first_derivative = np.gradient(relative_gaussian)
        # second order derivative
        second_derivative = np.gradient(first_derivative)
        # find the index and value of local maximum
        zero_points = np.where(np.isclose(second_derivative, 0.1, rtol=0.1, atol=0.2))[0]
        print('Zero points:', unique_elements[zero_points])
        local_minimum = relative_gaussian[zero_points]
        print('Local max:', local_minimum)
        max_local_minimum = np.array([i for i in local_minimum if i < -75]) # 75 meV
        print('Max local max:', max_local_minimum.max())
        # min_local_minimum = local_minimum.min()
        # print('Min local max:', min_local_minimum)
        min_value = relative_gaussian.min()
        print('Min value:', min_value)
        # 计算势垒高度
        barrier_height = max_local_minimum.max() - min_value
    else: # only one position
        print('One position')
        # barrier height == total max - total min
        max_value = relative_gaussian.max()
        min_value = relative_gaussian.min()
        # 计算势垒高度
        barrier_height = max_value - min_value
    return barrier_height

# calculate the free energy surface
def barrier1(middle_layer_total, temp_point, sgma=2):

    # 常数定义
    k_B = 8.617333262145e-5 # Boltzmann 常数，单位 eV/K
    # T = 300 # 温度，单位 K
    middle_process = np.around(middle_layer_total.flatten(), 2)
    num_middle = len(middle_process)
    # 计算每个原子在每个 bin 中的数量
    # hist, bin_edges = np.histogram(middle_process, bins=np.arange(-0.25, 1.25, 0.01), density=True)
    unique_elements, counts = np.unique(middle_process, return_counts=True)
    probabilities = counts / num_middle
    # 计算free energy based on the entropy
    free_energy = []
    for i in range(len(unique_elements)):
        free_energy.append(-k_B * temp_point * np.log(probabilities[i] + 1e-12)) # 添加一个小常数以避免 log(0)
    free_energy_array = np.array(free_energy)
    gaussian_free = gaussian_filter1d(free_energy_array, sigma=2)
    relative_free = 1000 * (free_energy_array - free_energy_array.max())
    relative_gaussian = 1000 * (gaussian_free - gaussian_free.max())

    return unique_elements, relative_free, relative_gaussian

# draw the free energy barrier
def draw_barrier(unique_elements, relative_free, relative_gaussian, barrier_height):
    print('Barrier height:', barrier_height, 'meV')
    # 绘制自由能图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(unique_elements, relative_free, label='Free Energy', color='blue', linewidth=2)
    ax.plot(unique_elements, relative_gaussian, label='Gaussian Free Energy', color='red', linewidth=2)
    ax.set_xlabel('Relative position', fontsize=22)
    ax.set_ylabel('Free Energy (meV)', fontsize=22)
    ax.legend()
    plt.savefig('free_energy.png')
    return barrier_height

############################################
#  Draw part
############################################

def colormap(color_type):
    # Choose a colormap 
    if color_type == 1:
        cmap = 'viridis'
    elif color_type == 2: 
        # Gradient colormap
        oranges_r = plt.get_cmap('Oranges_r')
        blues = plt.get_cmap('Blues')
        # 合并两个 colormap 的颜色
        combine = np.vstack((
            oranges_r(np.linspace(0, 1, 128)),
            blues(np.linspace(0, 1, 128))
        ))
        cmap = LinearSegmentedColormap.from_list('combined_colormap', combine)
    return cmap

def series_draw_picture_position(temps, type, output_name, start_time, end_time, min_range, max_range):
    cmap2 = 'inferno_r' 
    #'coolwarm'
    # cmap2 = colormap(2)

    #colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 40,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    # fig, axs= plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={'width_ratios': [5, 2]}) #, gridspec_kw={'height_ratios': [2, 1]})
    fig_num = int(len(temps))
    fig, axs = plt.subplots(fig_num, 1, sharex= True, figsize=(14, 15), dpi = 350) #10, 15
    norm = Normalize(vmin=min_range, vmax=max_range) #(vmin=-0.25, vmax=1.25)

    for count, temp in enumerate(temps):
        filepath = '../' + str(temp) + '/' + str(type) + '/position.npy'
        middle_layer_total = np.load(filepath)
        part_middle_layer_total = middle_layer_total[start_time:end_time]
        average_middle_layer = np.average(part_middle_layer_total)
        print('temperature', temp)
        print(average_middle_layer)

        # Real process
        steps = len(middle_layer_total)
        x = np.arange(steps)/10000
        y_number = middle_layer_total.shape[1]
        if count != (fig_num-1):
            for i in range(y_number):
            # Create line segments
                temp_points = middle_layer_total[:,i].flatten()
                points = np.array([x, temp_points]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # Create a LineCollection from the segments
                lc = LineCollection(segments, cmap=cmap2, norm=norm, alpha=0.6)
                lc.set_array(temp_points)
                lc.set_linewidth(1.5)
                axs[count].add_collection(lc)
            axs[count].set_xlim(0, 10)
            axs[count].set_ylim(min_range, max_range)
            axs[count].xaxis.set_major_locator(plt.MultipleLocator(2))
            axs[count].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            axs[count].spines['top'].set_linewidth(1.5)
            axs[count].spines['right'].set_linewidth(1.5)
            axs[count].spines['bottom'].set_linewidth(1.5)
            axs[count].spines['left'].set_linewidth(1.5)
            axs[count].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False, direction='in', width=1.5, length= 5)

        else:
            # draw the last ax
            for i in range(y_number):
            # Create line segments
                temp_points = middle_layer_total[:,i].flatten()
                points = np.array([x, temp_points]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # Create a LineCollection from the segments
                lc = LineCollection(segments, cmap=cmap2, norm=norm, alpha=0.6)
                lc.set_array(temp_points)
                lc.set_linewidth(1.5)
                axs[count].add_collection(lc)
            axs[count].set_xlim(0, 10)
            axs[count].set_ylim(min_range, max_range)
            axs[count].xaxis.set_major_locator(plt.MultipleLocator(2))
            axs[count].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            axs[count].spines['top'].set_linewidth(1.5)
            axs[count].spines['right'].set_linewidth(1.5)
            axs[count].spines['bottom'].set_linewidth(1.5)
            axs[count].spines['left'].set_linewidth(1.5)
            axs[count].set_xlabel('Steps (ps)')
            # axs[count].set_ylabel('Relative position', fontsize=28)
            axs[count].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, direction='in', width=1.5, length= 5)
            axs[count].xaxis.set_tick_params(width=1.5)
            axs[count].yaxis.set_tick_params(width=1.5)
    # fig.supxlabel('steps (ps)', fontsize=28, ha='center', va='center')
    fig.supylabel(t='${r}$$_{z}$', x=0.05, y=0.5, ha='center', va='center')
    # plt.subplots_adjust(hspace=0.2)
    
        # cb = fig.colorbar(lc, ax=ax, orientation='vertical')
        # cb.ax.yaxis.set_major_locator(plt.MultipleLocator(0.25)) # pass the information from the ax to the colorbar
        # cb.set_ticklabels([])
        # cb.ax.spines['top'].set_linewidth(1.5)
        # cb.ax.spines['right'].set_linewidth(1.5)
        # cb.ax.spines['bottom'].set_linewidth(1.5)
        # cb.ax.spines['left'].set_linewidth(1.5)
        # cb.ax.tick_params(width=1.5, length= 7)

    # bx = axs[1]
    # # draw the free energy surface
    # bx.plot(relative_free, unique_elements, label=r'E$_{b}$', color='blue', linewidth=2, alpha=0.75)
    # bx.plot(relative_gaussian, unique_elements, label=r'fitted E$_{b}$', color='lightblue', linewidth=2, alpha=0.75, linestyle='--')
    # bx.set_ylim(min_range, max_range)
    # bx.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    # bx.xaxis.set_major_locator(plt.MultipleLocator(50))
    # bx.spines['top'].set_linewidth(1.5)
    # bx.spines['right'].set_linewidth(1.5)
    # bx.spines['bottom'].set_linewidth(1.5)
    # bx.spines['left'].set_linewidth(1.5)
    # bx.tick_params(width=1.5, length= 7)
    # bx.xaxis.inverted = True
    # bx.set_xlabel('Free Energy (meV)')
    # bx.legend(loc = 'best', fontsize=22)
    # # if height != 0:
    # #     h_text = np.round(height, 1)
    # #     bx.text(0.3, 0.9, str(h_text) + ' meV', fontsize=22, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(output_name)
    return


def series_draw_picture_density(temps, type, output_name, start_time, end_time, min_range, max_range):
    # 选择 colormap
    barrier_max = -400
    #colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 38,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    # fig, axs= plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={'width_ratios': [5, 2]}) #, gridspec_kw={'height_ratios': [2, 1]})
    fig_num = int(len(temps))
    fig, bxs = plt.subplots(fig_num, 1, sharex= True, figsize=(7, 15), dpi = 350)
    ratio_y_x = np.abs(barrier_max)/1.5
    for count, temp in enumerate(temps):
        filepath = '../' + str(temp) + '/' + str(type) + '/position.npy'
        middle_layer_total = np.load(filepath)
        part_middle_layer_total = middle_layer_total[start_time:end_time]

        print('temperature', temp)
        unique_elements, relative_free, relative_gaussian= barrier1(part_middle_layer_total, temp, sgma=2)
        barrier_height = height_value(unique_elements, relative_gaussian, max_range, min_range)
        # barrier_height = relative_gaussian[len(unique_elements)//2] -  relative_gaussian.min()
        # print('Middle of energy surface:', unique_elements[len(unique_elements)//2])
        print('Barrier height:', barrier_height, 'meV')
        # Real process
        steps = len(middle_layer_total)
        x = np.arange(steps)/10000

        ### Note: bxs[0] has legend, bxs[1:] has no legend, bxs[-1] has xlabel
        if count == 0:
            bxs[count].plot(relative_free, unique_elements, label=r'${F}$$_{h}$', color='black', linewidth=3, alpha=0.85)
            bxs[count].plot(relative_gaussian, unique_elements, label=r'fitted ${F}$$_{h}$', color='cyan', linewidth=3, alpha=0.85, linestyle='--')
            bxs[count].set_ylim(min_range, max_range)
            bxs[count].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            bxs[count].xaxis.set_major_locator(plt.MultipleLocator(100))
            bxs[count].set_xlim(barrier_max, 0)
            bxs[count].spines['top'].set_linewidth(1.5)
            bxs[count].spines['right'].set_linewidth(1.5)
            bxs[count].spines['bottom'].set_linewidth(1.5)
            bxs[count].spines['left'].set_linewidth(1.5)
            bxs[count].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False, direction='in', width=1.5, length= 5)
            bxs[count].xaxis.inverted = True
            # bx.set_xlabel('Free Energy (meV)')
            bxs[count].legend(loc = 'best', fontsize=33)
            bxs[count].set_aspect(ratio_y_x)

        elif count != (fig_num-1) and count != 0:
            bxs[count].plot(relative_free, unique_elements, label=r'${F}$$_{h}$', color='black', linewidth=3, alpha=0.85)
            bxs[count].plot(relative_gaussian, unique_elements, label=r'fitted ${F}$$_{h}$', color='cyan', linewidth=3, alpha=0.85, linestyle='--')
            bxs[count].set_ylim(min_range, max_range)
            bxs[count].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            bxs[count].xaxis.set_major_locator(plt.MultipleLocator(100))
            bxs[count].set_xlim(barrier_max, 0)
            bxs[count].spines['top'].set_linewidth(1.5)
            bxs[count].spines['right'].set_linewidth(1.5)
            bxs[count].spines['bottom'].set_linewidth(1.5)
            bxs[count].spines['left'].set_linewidth(1.5)
            bxs[count].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False, direction='in', width=1.5, length= 5)
            bxs[count].xaxis.inverted = True
            # bx.set_xlabel('Free Energy (meV)')
            # bxs[count].legend(loc = 'best', fontsize=28)
            bxs[count].set_aspect(ratio_y_x)

        else:
            # draw the last bx
            bxs[count].plot(relative_free, unique_elements, label=r'${F}$$_{h}$', color='black', linewidth=3, alpha=0.85)
            bxs[count].plot(relative_gaussian, unique_elements, label=r'fitted ${F}$$_{h}$$', color='cyan', linewidth=3, alpha=0.85, linestyle='--')
            bxs[count].set_ylim(min_range, max_range)
            bxs[count].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            bxs[count].xaxis.set_major_locator(plt.MultipleLocator(100))
            bxs[count].set_xlim(barrier_max, 0)
            bxs[count].spines['top'].set_linewidth(1.5)
            bxs[count].spines['right'].set_linewidth(1.5)
            bxs[count].spines['bottom'].set_linewidth(1.5)
            bxs[count].spines['left'].set_linewidth(1.5)
            bxs[count].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, direction='in', width=1.5, length= 5)
            bxs[count].xaxis.set_tick_params(width=1.5, rotation=45)
            bxs[count].xaxis.inverted = True
            bxs[count].set_xlabel(r'${F}$$_{h}$ (meV)')
            # bxs[count].legend(loc = 'best', fontsize=28)
            bxs[count].set_aspect(ratio_y_x)
            # bxs[count].set_aspect('auto')

    # fig.supxlabel('steps (ps)', fontsize=28, ha='center', va='center')
    # fig.supxlabel(t='Free energy (meV)', x=0.07, y=0.5, fontsize=28, ha='center', va='center')

    # plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    plt.savefig(output_name)
    return   

if __name__ == '__main__':
    type = 'up_init'
    #type = 'mono'
    start_time1 = 1 #1 
    end_time1 = 100000 #100000
    start_time2 = 50000 # Using the last 5 ps to calculate the free energy surface
    end_time2 = 100000 #100000    
    outfilename1 = 'GR_up_jump.png'
    outfilename2 = 'GR_up_barriers.png'
    min_range, max_range = -0.25, 1.25
    #temps = [290, 370, 410, 500, 550]
    #temps = [ 310, 330, 350, 390, 450]
    temps = [290, 370, 450]
    #temps = [390, 410, 450, 500, 550]

    #temps = [290, 310, 330, 350, 370, 390, 410, 450, 500, 550]
    series_draw_picture_position(temps, type, outfilename1, start_time1, end_time1, min_range, max_range)
    series_draw_picture_density(temps, type, outfilename2, start_time2, end_time2, min_range, max_range)



