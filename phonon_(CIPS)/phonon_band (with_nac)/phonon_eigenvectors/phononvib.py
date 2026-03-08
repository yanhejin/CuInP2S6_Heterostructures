#!/usr/bin/env python

### a simple script to extract the vibrational vectors from band.yaml file

import os
import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import unicodedata
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
# from ase.io import read, write
# from ase import Atoms

############################################################
elements_sign = [
    'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 'Na', 'Mg', 'Al',
    'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I ', 'Xe', 'X ', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
]

elements_mass = [
    '1.00794    ','4.002602   ','6.941      ','9.0121831  ','10.811     ','12.0107    ',
    '14.0067    ','15.9994    ','18.99840316','20.1797    ','22.98976928','24.3050    ',
    '26.9815385 ','28.0855    ','30.97376199','32.065     ','35.453     ','39.948     ',
    '39.0983    ','40.078     ','44.955908  ','47.867     ','50.9415    ','51.9961    ',
    '54.938044  ','55.845     ','58.933194  ','58.6934    ','63.546     ','65.38      ',
    '69.723     ','72.64      ','74.921595  ','78.971     ','79.904     ','83.798     ',
    '85.4678    ','87.62      ','88.90584   ','91.224     ','92.90637   ','95.95      ',
    '98.9072    ','101.07     ','102.90550  ','106.42     ','107.8682   ','112.414    ',
    '114.818    ','118.710    ','121.760    ','127.60     ','126.90447  ','131.293    ',
    '132.9054519','137.327    ','138.90547  ','140.116    ','140.90766  ','144.242    ',
    '144.9      ','150.36     ','151.964    ','157.25     ','158.92535  ','162.500    ',
    '164.93033  ','167.259    ','168.93422  ','173.054    ','174.9668   ','178.49     ',
    '180.94788  ','183.84     ','186.207    ','190.23     ','192.217    ','195.084    ',
    '196.966569 ','200.59     ','204.3833   ','207.2      ','208.98040  ','208.9824   ',
    '209.9871   ','222.0176   ',    
]

############################################################

# read in the born charge from the vasp OUTCAR file


# read in the phonon vecotors and frequency at a q point
def read_vibvectors(file_name):
    bandfile = [line for line in open(file_name) if line.strip()]
    ln  = len(bandfile)

    for line in bandfile:
        if "natom" in line:
            atoms_num = int(line.split()[-1])
            break

    for line in bandfile:
        if "nqpoint" in line:
            total_q_num = int(line.split()[-1])
            break

    mass = []
    for i in range(0,ln):
            if "mass" in bandfile[i]:
                mass.append(float(bandfile[i].split()[-1]))
    
    frequency = []
    for i in range(0,ln):
            if "frequency" in bandfile[i]:
                frequency.append(float(bandfile[i].split()[-1]))

    total_freq_num = atoms_num * 3
    fre_array = np.array(frequency, dtype=float).reshape((total_q_num, total_freq_num))
    fre_array_real = fre_array[:, 3:] # The image freqency are neglected, corresbonding to the first three

    # Dim for vibvectors: q point, freq number, atom number 
    vib = []
    for i in range(0,ln):
        if '# atom' in bandfile[i]:
            for j in range(1,4):
                extract1  = bandfile[i+j].split()[2]
                extract2  = bandfile[i+j].split()[3]
                real_part = float(extract1.replace(',', ''))
                imag_part = float(extract2) 
                vec_temp  = complex(real_part, imag_part)
                vib.append(vec_temp)
            # break
    vib_array = np.array(vib, dtype=complex).reshape((-1, total_freq_num, atoms_num, 3))
    vib_array_real = vib_array[:,3:,:,:] # The image freqency are neglected, corresbonding to the first three

    return vib_array_real, fre_array_real

# extract certain eigenvector of point q (q start from number 1, freq start from number 1)
def extract(q_num, vib_array_real):
    atoms_num = vib_array_real.shape[2]
    certain_vib = vib_array_real[q_num - 1]
    certain_vib_array = certain_vib.reshape((-1, atoms_num, 3))
    return certain_vib_array

############################################################
# Generate th xsf files

def generate_LO_vib_axsf(mode_series,polar_vector,frequencies_cm1,position , method, start_vib, final_vib, freq_range):
    # S-intensity denotes the polarization of the crystall  
    vector_num = int(frequencies_cm1.shape[0])
    atom_num   = int(vector_num/3) + 1
    #print(atom_num)
    f2 = open(position,"r")
    f2_original = f2.readlines()
    f2.close()
    f2_position = [line.split() for line in f2_original[8:(8+atom_num)]]
    f2_position_d = np.array(f2_position, dtype=float).reshape((-1, 3))
    f2_lattice = [line.split() for line in f2_original[2:5]]
    f2_lattice = np.array(f2_lattice, dtype=float).reshape((3, 3))
    f2_position_c = convert_to_cartisian(f2_position_d,f2_lattice,atom_num)

    # generate the element series for lable
    ElementNames,ElementTotal,AtomNumbers,Atomtotal = element_information(position)
    #example: For Sb2S3, ElementNames = [Sb,S]; ElementTotal = 2, AtomNumbers = [24,36], Atomtotal = 60    
    lable_series = []   
    for ele_sequent in range(0,ElementTotal):
        for j in range(0,AtomNumbers[ele_sequent]):
            lable_series.append(ElementNames[ele_sequent])
    print(lable_series)
    #lable_series is the element series [Sb, Sb, Sb ... S, S, S], this will be used for lable

    if method == "total":
        for i in range(vector_num):
            vector_temp = polar_vector[i]
            vector_temp = np.array(vector_temp, dtype=float).reshape((-1, 3))
            write_xsf(i+1,frequencies_cm1[i],lable_series,f2_lattice,f2_position_c,vector_temp,atom_num)

    elif method == "part":
        vibstart = int(start_vib) - 1
        vibfinal = int(final_vib) - 1
        for i in range(vibstart,vibfinal):
            vector_temp = polar_vector[i]
            vector_temp = np.array(vector_temp, dtype=float).reshape((-1, 3))
            write_xsf(i+1,frequencies_cm1[i],lable_series,f2_lattice,f2_position_c,vector_temp,atom_num)

    elif method == "range":
        #force the out of range terms equles to zero
        peak = []
        for i in range(vector_num):
            if freq_range[0] < frequencies_cm1[i] and frequencies_cm1[i] < freq_range[-1]:
                peak.append(i)     ### make the s-intensity of our of range modes equels to zero
                                    ### this will convinent the sort process
        peak = np.array(peak, dtype=int)
        print(peak)
        count = 1
        for i in peak:
            vector_temp = polar_vector[i]
            vector_temp = np.array(vector_temp, dtype=float).reshape((-1, 3))
            write_xsf(i+1,frequencies_cm1[i],lable_series,f2_lattice,f2_position_c,vector_temp,atom_num)
            #write_xsf_intensity(count,i+1,frequencies_cm1[i],lable_series,f2_lattice,f2_position_c,vector_temp,atom_num)
            count +=1                

    elif method == "intensity":
        #extract the polarization with strong S intensity, the large mode S intensity also means the large LO spliting
        count = 1
        for i in mode_series:
            vector_temp = polar_vector[i]
            vector_temp = np.array(vector_temp, dtype=float).reshape((-1, 3))
            write_xsf(i+1,frequencies_cm1[i],lable_series,f2_lattice,f2_position_c,vector_temp,atom_num)
            #write_xsf_intensity(count,i+1,frequencies_cm1[i],lable_series,f2_lattice,f2_position_c,vector_temp,atom_num)
            count +=1

def convert_to_cartisian(position_d,lattice_vector,natoms):    
    position_c = np.empty([natoms,3], dtype=float)
    for i in range (0,natoms):
        for j in range(0,3):
            position_c[i][j] = position_d[i][0]*lattice_vector[0][j]+position_d[i][1]*lattice_vector[1][j]+position_d[i][2]*lattice_vector[2][j]
    return position_c

def element_information(phonon_POSCAR):
    tensor_file = open(phonon_POSCAR,"r")
    f_0 = tensor_file.readlines()
    tensor_file.close()

    ElementNames = f_0[5].split()
    ElementTotal = len(list(ElementNames))
    AtomNumbers = np.array([int(x) for x in f_0[6].split()], dtype=int)
    totalatom = AtomNumbers.sum()
    return ElementNames,ElementTotal,AtomNumbers,totalatom
    #example: For Sb2S3, ElementNames = [Sb,S]; ElementTotal = 2, AtomNumbers = [24,36], totalatom = 60 

def write_xsf(name,fre,lable,real_lattice,real_position,vector,atom_num):
    total_vector = np.hstack((real_position, vector))
    # Here, the different 
    with open('withnac_mode{:04d}fre{:.2f}.xsf'.format(name, fre), 'w') as out_phonon:
        line = "CRYSTAL\n"
        line += "PRIMVEC\n"
        line += '\n'.join([
            ' '.join(['%21.16f' % a for a in real_lattice[ii]])
            for ii in range(3)
        ])
        line += "\nPRIMCOORD\n"
        line += "{:3d} {:d}\n".format(atom_num, 1)
        line += '\n'.join([
            '{:3s}'.format(lable[ii]) +
            ' '.join(['%21.16f' % a for a in total_vector[ii]])
            for ii in range(atom_num)
        ])
        out_phonon.write(line)

############################################################

if __name__ == '__main__':
    
    ### Read from band.yaml
    # if not os.path.isfile('atom_born.npy'):    
    #     colors = ['dimgrey', 'purple','darkcyan','blue','lightpink', '#FAC748']  ###Choose the color you want
    #     modes,omegas = seperate_vibmode_from_Outcar('phonon_OUTCAR', include_imag=True) #True
    #     atom_born = borncharge_from_Outcar('born_OUTCAR')
    #     np.save('omega.npy', omegas)
    #     np.save('modes.npy', modes)    
    #     np.save('atom_born.npy', atom_born)
    # else:

    #Basic test1
    file_name1 = '../withnac/band.yaml'
    vib_total, fre_total = read_vibvectors(file_name1)
    # print(vib_total.shape)
    # print(fre_total.shape)
    # print(fre_total)
    q_num = 1
    ini_vib = extract(q_num, vib_total)
    fre = fre_total[q_num-1]
    print(ini_vib.shape)

    THz_to_cmm1 = 33.356
    fre_cm = fre * THz_to_cmm1
    position = './POSCAR'
    vibinfo= ini_vib.real
     
    range_array = np.array([0,375]) # freq range for generation 1400, 1500
    freq_range = range_array
    method = "range"
    mode_series = []
    start_vib = 0
    final_vib = 0
    
    #generate_LO_vib_axsf(mode_series= peak[::-1],inf2= 'polor_lo.npy',fre = 'omega_lo.npy',position = 'phonon_POSCAR', method = "intensity")
    ### draw the NAC considered phonon modes
    generate_LO_vib_axsf(mode_series, vibinfo, fre_cm, position, method, start_vib, final_vib, freq_range)


      
    #generate_LO_vib_axsf(mode_series,inf2,fre,position , method, start_vib, final_vib, freq_range)
