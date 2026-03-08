import sys
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import os
#import ase
#from ase.io import read
#from ase.data import atomic_numbers, covalent_radii
# from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d, griddata
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
        zero_points = np.where(np.isclose(second_derivative, 0.05, rtol=0.05, atol=0.1))[0]
        # zero_points = np.where(np.isclose(second_derivative, 0.01, rtol=0.01, atol=0.05))[0]
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

# calculate the free energy surface with intropolation method
def barrier1_intropo(middle_layer_total, temp_point, min_range, max_range, sgma=2):

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
    # 使用插值方法
    interp_func = interp1d(unique_elements, free_energy_array, kind='cubic', fill_value='extrapolate')
    # 生成更细的 x 轴数据
    x_new = np.linspace(unique_elements.min(), unique_elements.max(), num=1000)
    # print(unique_elements.min(), unique_elements.max())
    
    # x_new = np.linspace(min_range, max_range, num=1000) # ! Notice, this line leads wrong result
    # 计算插值后的自由能
    free_energy_interp = interp_func(x_new)
    # 计算高斯滤波后的自由能
    gaussian_free = gaussian_filter1d(free_energy_interp, sigma=sgma)
    # 计算相对自由能
    relative_free = 1000 * (free_energy_interp - free_energy_interp.max())
    relative_gaussian = 1000 * (gaussian_free - gaussian_free.max())
    # 计算高斯滤波后的自由能
    # gaussian_free = gaussian_filter1d(free_energy_array, sigma=2)
    # relative_free = 1000 * (free_energy_array - free_energy_array.max())
    # relative_gaussian = 1000 * (gaussian_free - gaussian_free.max())

    return x_new, relative_free, relative_gaussian


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
    # 选择 colormap
    cmap2 = colormap(2)

    #colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 32,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    # fig, axs= plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={'width_ratios': [5, 2]}) #, gridspec_kw={'height_ratios': [2, 1]})
    fig_num = int(len(temps))
    fig, axs = plt.subplots(fig_num, 1, sharex= True, figsize=(17, 25), dpi = 350) #10, 15
    norm = Normalize(vmin=min_range, vmax=max_range) #(vmin=-0.25, vmax=1.25)

    for count, temp in enumerate(temps):
        filepath = './' + str(temp) + '/' + str(type) + '/position.npy'
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
    fig.supylabel(t='Relative position', x=0.05, y=0.5, ha='center', va='center')
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
    cmap2 = colormap(2)
    barrier_max = -400
    #colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 32,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    # fig, axs= plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={'width_ratios': [5, 2]}) #, gridspec_kw={'height_ratios': [2, 1]})
    fig_num = int(len(temps))
    fig, bxs = plt.subplots(fig_num, 1, sharex= True, figsize=(7, 25), dpi = 350)
    ratio_y_x = np.abs(barrier_max)/1.5
    for count, temp in enumerate(temps):
        filepath = './' + str(temp) + '/' + str(type) + '/position.npy'
        middle_layer_total = np.load(filepath)
        part_middle_layer_total = middle_layer_total[start_time:end_time]

        print('temperature', temp)
        unique_elements, relative_free, relative_gaussian= barrier1(part_middle_layer_total, temp, sgma=2)
        # unique_elements, relative_free, relative_gaussian = barrier1_intropo(part_middle_layer_total, temp, sgma=2)
        barrier_height = height_value(unique_elements, relative_gaussian, max_range, min_range)
        # barrier_height = relative_gaussian[len(unique_elements)//2] -  relative_gaussian.min()
        # print('Middle of energy surface:', unique_elements[len(unique_elements)//2])
        print('Current shape',unique_elements.shape)
        print('Barrier height:', barrier_height, 'meV')
        # Real process
        steps = len(middle_layer_total)
        x = np.arange(steps)/10000

        ### Note: bxs[0] has legend, bxs[1:] has no legend, bxs[-1] has xlabel
        if count == 0:
            bxs[count].plot(relative_free, unique_elements, label=r'E$_{h}$', color='blue', linewidth=2, alpha=0.75)
            bxs[count].plot(relative_gaussian, unique_elements, label=r'fitted E$_{h}$', color='lightblue', linewidth=2, alpha=0.75, linestyle='--')
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
            bxs[count].legend(loc = 'best', fontsize=30)
            bxs[count].set_aspect(ratio_y_x)

        elif count != (fig_num-1) and count != 0:
            bxs[count].plot(relative_free, unique_elements, label=r'E$_{h}$', color='blue', linewidth=2, alpha=0.75)
            bxs[count].plot(relative_gaussian, unique_elements, label=r'fitted E$_{h}$', color='lightblue', linewidth=2, alpha=0.75, linestyle='--')
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
            bxs[count].plot(relative_free, unique_elements, label=r'E$_{h}$', color='blue', linewidth=2, alpha=0.75)
            bxs[count].plot(relative_gaussian, unique_elements, label=r'fitted E$_{h}$', color='lightblue', linewidth=2, alpha=0.75, linestyle='--')
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
            bxs[count].set_xlabel('Free Energy (meV)')
            # bxs[count].legend(loc = 'best', fontsize=28)
            bxs[count].set_aspect(ratio_y_x)
            # bxs[count].set_aspect('auto')

    # fig.supxlabel('steps (ps)', fontsize=28, ha='center', va='center')
    # fig.supxlabel(t='Free energy (meV)', x=0.07, y=0.5, fontsize=28, ha='center', va='center')

    # plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    plt.savefig(output_name)
    return   

def series_draw_3D_waterfall(temps, type, output_name, start_time, end_time, min_range, max_range): 
    # 选择 colormap
    if not os.path.exists('./draw_up_x.npy'):
        draw_x = np.zeros((len(temps), 1000), dtype=float)
        draw_y = np.zeros((len(temps), 1000), dtype=float)
        for count, temp in enumerate(temps):
            filepath = './' + str(temp) + '/' + str(type) + '/position.npy'
            middle_layer_total = np.load(filepath)
            part_middle_layer_total = middle_layer_total[start_time:end_time]

            print('temperature', temp)
            # unique_elements, relative_free, relative_gaussian= barrier1(part_middle_layer_total, temp, min_range, max_range, sgma=2)
            unique_elements, relative_free, relative_gaussian = barrier1_intropo(part_middle_layer_total, temp, min_range, max_range, sgma=2)
            # barrier_height = height_value(unique_elements, relative_gaussian, max_range, min_range)
            # barrier_height = relative_gaussian[len(unique_elements)//2] -  relative_gaussian.min()
            # print('Middle of energy surface:', unique_elements[len(unique_elements)//2])
            print('Current shape',unique_elements.shape)
            draw_x[count] = unique_elements
            draw_y[count] = relative_gaussian
        np.save('draw_up_x.npy', draw_x)
        np.save('draw_up_y.npy', draw_y)
    else:
        draw_x = np.load('./draw_up_x.npy')
        draw_y = np.load('./draw_up_y.npy')
    # print('Draw x shape:', draw_x.shape)
    # print('Draw y shape:', draw_y.shape)

    barrier_max = -400
    #colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 32,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    # fig, axs= plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={'width_ratios': [5, 2]}) #, gridspec_kw={'height_ratios': [2, 1]})

    # ratio_y_x = np.abs(barrier_max)/1.5


    # 创建 3D 图形
    fig = plt.figure(figsize=(12, 6), dpi = 350)
    ax = fig.add_subplot(111, projection='3d')

    temp_points = np.array(temps)  # 温度索引轴
    num_points = draw_x.shape[1]  # 每个温度点的 Raman shift 数量
    # 绘制每个温度点的 Raman 曲线
    for i in range(len(temps)):
        ax.plot(draw_x[i], [temp_points[i]] * num_points, draw_y[i], label=f'Temp {temps[i]}')

    # 设置标签和标题
    ax.set_xlabel('Energy (meV)')
    ax.set_ylabel('Temperature')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Waterfall Plot of surface Data')
    plt.tight_layout()
    plt.savefig(output_name)
    return

def series_draw_3D_surface(temps, output_name):
    draw_x = np.load('./draw_up_x.npy')
    draw_y = np.load('./draw_up_y.npy')

    temperature_indices = np.array(temps)  # 温度索引轴
    # 创建网格点
    X, Y = np.meshgrid(draw_x[-1], temperature_indices)
    Z = draw_y

    # 插值到更细的网格以形成平滑曲面
    xi = np.linspace(draw_x[-1].min(), draw_x[-1].max(), 1200)
    yi = np.linspace(temperature_indices.min(), temperature_indices.max(), 100)
    XI, YI = np.meshgrid(xi, yi)

    # 将原始数据点展开为一维数组
    points = np.array([X.flatten(), Y.flatten()]).T
    values = Z.flatten()

    # 使用 griddata 进行插值
    ZI = griddata(points, values, (XI, YI), method='cubic')


    # 绘制三维曲面图
    font = {'family': 'Arial', 'size': 16,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    cs = ax.plot_surface(XI, YI, ZI, cmap=cm.coolwarm, edgecolor='none', rstride = 1, cstride = 1,alpha = 1, antialiased=False)
    
    #'rainbow'
    #'Spectral_r'
    #'viridis'
    #'icefire'

    # 设置标签和标题
    ax.invert_xaxis()  # 反转 x 轴
    ax.set_xlabel('Relative position', labelpad=15, fontsize=20)
    ax.set_ylabel('Temperature (K)', labelpad=20, fontsize=20)
    ax.tick_params(axis='x', pad=5)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=15)


    # ax.set_zlabel('Energy (meV)')
    # ax.set_title('Interpolated Potential Energy Surface', labelpad=10)
    ax.view_init(elev=25, azim=80)  # elev是仰角，azim是方位角
    #ax.view_init(elev=15, azim=90)
    # 去除背景和网格
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    # ax.grid(False)
    # ax.set_facecolor('white')  # 设置背景颜色为白色
    # ax.grid(True, linestyle='--')

    # ax.patch.set_facecolor("white")
    # ax.spines['left'].set_linestyle('--')
    # # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linestyle('--')
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.patch.set_alpha(0.2)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.4, aspect=8, pad=0.015)
    cbar.set_label('Free energy (meV)', fontsize=20, labelpad=6)

    plt.tight_layout()
    plt.savefig(output_name)
    plt.show()
    return


if __name__ == '__main__':
    type = 'up_init'
    # type = 'mono'
    start_time1 = 1 #1 
    end_time1 = 100000 #100000
    start_time2 = 50000 #1 
    end_time2 = 100000 #100000
    filepath = '/' + str(type) + 'position.npy' # './' + str(temp) + str(type) + 'position.npy'
    # outfilename1 = 'AAA1_up.png'
    outfilename2 = 'AAA2_up.png'
    min_range, max_range = -0.3, 1.3
    #temps = [290, 370, 410, 500, 550]
    #temps = [ 310, 330, 350, 390, 450]

    
    # temps = [290, 310, 330, 350, 370]
    # temps = [390, 410, 450, 500, 550]
    temps = [290, 310, 330, 350, 370, 390, 410, 450, 500, 550]
    # series_draw_picture_position(temps, type, outfilename1, start_time1, end_time1, min_range, max_range)
    # series_draw_picture_density(temps, type, outfilename2, start_time2, end_time2, min_range, max_range)
    # series_draw_3D_waterfall(temps, type, outfilename1, start_time2, end_time2, min_range, max_range)
    series_draw_3D_surface(temps, outfilename2)



