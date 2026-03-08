import os
import sys
import numpy as np
import pandas as pd
import ase
from ase.io import read
from ase.data import atomic_numbers, covalent_radii
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

### process the position data
def structure_process(filename, kind1):
    md_traj = read(str(filename), index=':')
    steps = len(md_traj)
    print(steps)
    possition_total = []
    for i in range(steps):
        temp_traj = md_traj[i]
        possition_temp = process_atoms(kind1, temp_traj)
        possition_total.append(possition_temp)
    possition_total = np.array(possition_total).reshape(steps, -1, 3)

    return possition_total

def process_atoms(kind1, temp_traj):
    # here the kind1 represents the atom type of the Cu layer, kind2 represents the atom type of the S layer
    # temp_traj is the trajectory of the trj[i]
    kind1_copy = temp_traj.copy()
    del kind1_copy[[atom.index for atom in kind1_copy if atom.symbol !=kind1]]
    kind1_cartisian = kind1_copy.get_positions()

    return kind1_cartisian

### calculate the velocity
def velocity(atoms_position,mdstep):
    time = atoms_position.shape[0]
    natoms = atoms_position.shape[1]
    velocities = []
    #velocities = np.empty([time-1,natoms],dtype=float)
    ###开始记录原子的速度
    for i in range(1,time):
        for j in range(natoms):
            # dleta_x = atoms[i][j][0] - atoms[i-1][j][0]
            # dleta_y = atoms[i][j][1] - atoms[i-1][j][1]
            # dleta_z = atoms[i][j][2] - atoms[i-1][j][2]
            #velocity_temp = np.average(np.abs(atoms[i][j] - atoms[i-1][j]))/mdstep
            velocity_temp = np.average(atoms_position[i][j] - atoms_position[i-1][j])/mdstep
            velocities.append(velocity_temp)
    velocities = np.reshape(velocities,[time-1,natoms])
    average_velocity = np.average(velocities, axis = 1)
    return average_velocity

def find_pdos(v_all, Nc, dt, omega):    #Calculate the vacf from velocity data
    Nf = v_all.shape[0]                 # number of frames
    M  = Nf - Nc                        # number of time origins for time average
    vacf = np.zeros(Nc)                 # the velocity autocorrelation function (VACF)
    for nc in range(Nc):                # loop over the correlation steps
        ratio = (nc+1)/Nc * 100    
        print("Calculate PDOS Progress %s%%" %ratio)
        for m in range(M+1):            # loop over the time origins
            delta = np.sum(v_all[m + 0]*v_all[m + nc])
            # print(delta)
            vacf[nc] = vacf[nc] + delta
    vacf = vacf / vacf[0]                                       # normalize the VACF
    vacf_output = vacf                                          # copy the VACF before modifying it
    vacf = vacf*(np.cos(np.pi*np.arange(Nc)/Nc)+1)*0.5          # window function
    vacf = vacf*np.append(np.ones(1), 2*np.ones(Nc-1))/np.pi    # C(t) = C(-t)
    pdos = np.zeros(len(omega))                                 # the phonon density of states (PDOS)
    for n in range(len(omega)):                                 # Discrete cosine transform
        pdos[n] = dt * sum(vacf * np.cos(omega[n] * np.arange(Nc) * dt))
    return(vacf_output, pdos)


### Draw the picture
def draw_picture1(t, vacf, pdos, nu, name):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(t, vacf, linewidth = 2, color="C1")
    plt.xticks(fontsize = 14)
    plt.xlabel('Correlation Time (ps)', fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylabel('Normalized VACF', fontsize = 14)

    plt.subplot(1, 2, 2)
    plt.plot(nu, pdos, linewidth = 2, color="C4")
    plt.xlim(0,1000)
    #plt.ylim(0,0.02)
    plt.xticks(fontsize = 14)
    plt.xlabel('Wavenumber (cm-1)', fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylabel('PDOS', fontsize = 14)

    plt.subplots_adjust(wspace=0.3)
    name_out = 'compare' + str(name) + '.png'
    plt.savefig(name_out, bbox_inches='tight')

def draw_picture3(fft_x, fft_y, output_name):
    colors = ['#708090','#7C1A97','#BA55D3','#87447A','#63D07F','#9A32CD','#D2691E','#52535B','red','blue','#FAC748','pink', 'blue','#A7A5A0',  'dimgrey', 'purple', 'goldenrod']  ###Choose the color you want
    font = {'family': 'Arial', 'size': 22,'style': 'normal','weight': 'normal',} # 'weight': 'normal'
    plt.rc('font', **font)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(fft_x, fft_y, linewidth=1.5)
    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(0, 300)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    plt.title(output_name)
    plt.savefig(str(output_name))


if __name__ == '__main__':
    temp = sys.argv[1] # 290
    temp2 = sys.argv[2] # mono
    filepath = '../../' + str(temp) + '/' + str(temp2) + '/velocity.npy'
    start_time = 1
    end_time = 50000
    gap = 10000
    # start_time = 50000
    # end_time = 55000 

    Nc = 1200  # 3000 number of correlation steps (a larger number gives a finer resolution)
               # number of sample points
    dt             = 0.001
    omega          = np.arange(1, 1000, 0.5)
    t = np.arange(Nc)*dt

    ### transfer
    nu = (33.356*omega) / (2*np.pi)     # omega = 2 * pi * nu, while nu is the frequency range
                                        # 1 THz = 4.136 meV = 33.356 cm−1
    cut_frequency = 400
    cut_nu = nu[nu < cut_frequency]

### check purpose
    # vacf_total = []
    # pdos_toal = []
    # for i in range(2):
    #     name1 = str(i) + '_vacf.npy'
    #     name2 = str(i) + '_dos.npy'
    #     vacf_temp = np.load(name1)
    #     pdos_temp = np.load(name2)
    #     vacf_total.append(vacf_temp)
    #     pdos_toal.append(pdos_temp)
    # vacf_output1 = np.array(vacf_total).reshape(2, -1)
    # pdos1 = np.array(pdos_toal).reshape(2, -1)
    # print(vacf_output1.shape)
    # print(pdos1.shape)
    # print(vacf_output1)
    # print(pdos1)

### pdos calculation
    if not os.path.exists('0_dos.npy'):
        Cu_velocity = np.load(filepath)
        for i in range(5):
            temp_num = i * gap
            part_Cu = Cu_velocity[start_time + temp_num:end_time + temp_num]
            vacf_output1, pdos1 = find_pdos(part_Cu, Nc, dt, omega)    # Call the function and calculate the vacf and pdos
            name1 = str(i) + '_vacf.npy'
            name2 = str(i) + '_dos.npy'
            np.save(name1, vacf_output1)
            np.save(name2, pdos1)
    else:
        print('The file already exists!')
    # else:
    #     vacf_total = []
    #     pdos_toal = []
    #     for i in range(5):
    #         name1 = str(i) + '_vacf.npy'
    #         name2 = str(i) + '_dos.npy'
    #         vacf_temp = np.load(name1)
    #         pdos_temp = np.load(name2)
    #         vacf_total.append(vacf_temp)
    #         pdos_toal.append(pdos_temp)
    #     vacf_output1 = np.array(vacf_total).reshape(10, -1)
    #     pdos1 = np.array(pdos_toal).reshape(10, -1)
    # print(vacf_output1.shape)
    # t_list = t.tolist()
    # fre_num = len(cut_nu)
    # cut_nu_list = cut_nu.tolist()
    # df1 = pd.DataFrame(vacf_output1.T)
    # df1.columns = ['vacf1', 'vacf2', 'vacf3', 'vacf4', 'vacf5', 'vacf6', 'vacf7', 'vacf8', 'vacf9', 'vacf10'] 
    # df1.index = t_list
    # df2 = pd.DataFrame(pdos1[:,0:fre_num].T)
    # df2.columns = ['pdos1', 'pdos2', 'pdos3', 'pdos4', 'pdos5', 'pdos6', 'pdos7', 'pdos8', 'pdos9', 'pdos10']
    # df2.index = cut_nu_list
    # df1.to_csv('vacf.csv')
    # df2.to_csv('pdos.csv')


    
    # print(vacf_output1.shape)
    # print(pdos1.shape)

    # draw_picture1(t, vacf_output1, pdos1, nu, output_name1)
    # draw_picture3(nu, pdos1, output_name1)


        # with open("test.csv","w",newline ='') as csvfile:
        #     for i in pdos:
        #         csvfile.write(str(i))
        #         csvfile.write('\n')
