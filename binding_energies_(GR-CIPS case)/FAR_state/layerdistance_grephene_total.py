#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from ase import Atoms
from ase.io import vasp
from ase.constraints import FixScaled

def get_parser():
    parser = argparse.ArgumentParser(
        description='Input the file path')
    parser.add_argument('-dics')
    return parser

def element_info(inf):
    '''
    extract coordinates from POSCAR files.
    
    Input arguments:
        inf:        location of POSCAR
        direct:     coordinates in fractional or cartesian
    '''

    inp = open(inf).readlines()
    ElementNames = inp[5].split()
    ElementTotal = len(list(ElementNames))
    ElementNumbers = np.array([int(x) for x in inp[6].split()], dtype=int)
    cell = np.array([line.split() for line in inp[2:5]], dtype=float)
    Natoms = ElementNumbers.sum()
    return ElementNames , ElementTotal, ElementNumbers
    # Example: [Sb,S] [2], [2, 3] for Sb2S3
    ### Record
    #ASE read methods
    # ElementNames = geo0.get_chemical_symbols()
    # ElemenSimble = geo0.numbers
    # Simble, ElementNumbers = np.unique(np.array(ElemenSimble), return_counts= True)
    # ElementTotal = ElementNumbers.shape[0]
    # Natoms = sum(ElementNumbers)

def distance(series1, series2, geo0, pc0):
    position1 = 0
    position2 = 0
    atom_num1 = len(series1)
    atom_num2 = len(series2)
    for i in series1:
        position1 += pc0[i,-1]
    for j in series2:
        position2 += pc0[j,-1]
    average_position1 = position1/atom_num1
    average_position2 = position2/atom_num2
    diff = average_position1 - average_position2
    diff = diff * geo0.cell[2][2]
    return np.abs(diff)

def find_series(ElementNames , ElementTotal, ElementNumbers, pc0, lower_lim, upper_lim,  kind):   
    # print(ElementNames)
    # print(ElementNumbers)
    # print(ElementTotal)
    # determine the element number
    for i in range(ElementTotal):
        specified_name = 0
        if str(kind) == ElementNames[i]:
            specified_name = i
            break
    # print(specified_name)

    # determine the atom number
    start_number = 0
    end_number = 0
    if specified_name == 0:
        start_number = 0
        end_number = ElementNumbers[0]
    else:
        for j in range(specified_name):
           start_number += ElementNumbers[j]
        end_number = start_number + ElementNumbers[specified_name]
    # print(start_number)
    # print(end_number)

    # determine the correct atom series
    series = []
    for k in range(start_number, end_number):
        if pc0[k,-1] <= upper_lim and pc0[k,-1] >= lower_lim:
            series.append(k)
    return series

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    temp_dirctory = args.dics
    dics = './' + str(temp_dirctory)
    file1 = dics + '/total/CONTCAR'
    file2 = dics + '/total/OUTCAR'
    file3 = dics + '/lower/OUTCAR'
    file4 = dics + '/upper/OUTCAR'

    kind1 = 'S'
    lower_lim1, upper_lim1 = 0.15, 0.25
    kind2 = 'C'
    lower_lim2, upper_lim2 = 0.25, 0.9
    # kind3 = 'N'
    # lower_lim3, upper_lim3 = 0.25, 0.9
    kind4 = 'S'
    lower_lim4, upper_lim4 = 0.02, 0.15
    # load the starting geometry
    geo0 = vasp.read_vasp(file1)
    ElementNames , ElementTotal, ElementNumbers = element_info(file1)
    pc0  = geo0.get_scaled_positions().copy() # Direct position
    

    series_s = find_series(ElementNames, ElementTotal, ElementNumbers, pc0, lower_lim1, upper_lim1, kind1)
    series_c = find_series(ElementNames, ElementTotal, ElementNumbers, pc0, lower_lim2, upper_lim2, kind2)
    # series_n = find_series(ElementNames, ElementTotal, ElementNumbers, pc0, lower_lim3, upper_lim3, kind3)
    series_s2 = find_series(ElementNames, ElementTotal, ElementNumbers, pc0, lower_lim4, upper_lim4, kind4)
    # print(series_s)
    # print(series_c)
    # print(series_n)
    # print(series_s2)

    distance_s_c = distance(series_s, series_c, geo0, pc0)
    # distance_s_n = distance(series_s, series_n, geo0, pc0)
    # distnace_average = (distance_s_c + distance_s_n)/2
    distance_s_s2 = distance(series_s, series_s2, geo0, pc0)
    
    result0 =  vasp.read_vasp_out(file2)
    result1 =  vasp.read_vasp_out(file3)
    result2 =  vasp.read_vasp_out(file4)

    energy0 = result0.get_total_energy()
    energy1 = result1.get_total_energy()
    energy2 = result2.get_total_energy()
    print("%s %.5f %.5f %.6f %.6f %.6f" %(temp_dirctory, distance_s_c, distance_s_s2, energy0, energy1, energy2))
    #print("The layer distance is %.5f angstrom"%distnace_average)
    #print("The layer thickness of CuInP2S6 is %.5f angstrom"%distance_s_s2)




