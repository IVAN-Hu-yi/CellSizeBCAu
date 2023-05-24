# Helper functions

## intermediate values calcualtions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import permutation_test
from scipy.integrate import solve_ivp
from parametres import Paras
from utilities import *
from initialisation import *
from odes import odes_scale_size
import matplotlib.pyplot as plt
from size_scaled_func import *
import datetime as dt
from simulation_func import *
import pandas as pd
import pickle as pkl
import os
import seaborn as sns
import scipy.stats as stats
import networkx as nx

def vin(p, R, Rhalf, vmax=None, type=2):
    '''Calcualte Vin for each species

    Args:
        p (N by M matrix): preference 
        R (M by 1 matrix): resource concentration
        Rhalf ( float ): half-velocity constant
        vmax (N by M matrix, optional): maximum uptake rate. Defaults to None.
        type (int, optional): Type I or Type II functional response. Defaults to 2.

    Returns:
       N by 1 vector: Vin for each species
    '''
    # calculate uptake
    uptake = p * R.T[:, np.newaxis]
    uptake = uptake.reshape(p.shape)

    if type==2: # Monod function
        if vmax is None:
            print('Vmax required for monod function')
        else:
            return vmax*((uptake)/(Rhalf + uptake))

    elif type==1: # linear
        return uptake
    
def vgrow(vin, leakage):
    '''calculate resource flow needed for growth

    Args:
        vin (np.array): N*M matrix for resource inflow
        leakage (np.array): M*1 matrix -- leakage fraction

    Returns:
        N*1 vector: resource flow needed for growth

    '''
    return vin @ (1-leakage)

def vout(vin, leakage, D):
     
    '''caculate vout
    
    Args:
        vin (np.array): N*M matrix for resource inflow
        leakage (np.array): M*1 matrix -- leakage fraction
        D (np.array): N*M matrix for conversion efficiency

    Returns:
        np.array: N*M matrix
    '''

    vout = (vin @ D)* leakage.T[:, np.newaxis]
    vout = vout.reshape(vin.shape)
    return vout

### Results extractions

def extract_Ct_single_assembly(data):
    
    '''transform to a 2D matrix

    Args:
        data (np.array): N*tstep matrix from solve_ivp results

    Returns:
        _type_: N*tstep
    '''

    N, tsteps = data.shape
    Cts_single = np.empty((N, tsteps))
    for i in range(tsteps):
        Cts_single[:, i] = data[:, i]
    return Cts_single

def relative_abundance(data):
     '''transform them into relative abundance data

    Args:
        data (np.array): N*tstep matrix from solve_ivp results

    Returns:
        _type_: _description_
    '''
     data = extract_Ct_single_assembly(data)
     return data/np.sum(data, axis=0)

def extract_Ct_multiple(data):

    '''transform raw abundance data into a single N*tstep*(num of assemblies)
    
    Args:
        data (list): lists of assembly results 
    
    Returns:
        np.array : N*tstep*(num of assemblies)
    '''

    num = len(data)
    N, tstep = data[0].shape
    Cts_multiple = np.empty((N, tstep, num))
    for i in range(num):
        Cts_multiple[:, :, i] = extract_Ct_single_assembly(data[i])
    return Cts_multiple

#### Create Assocation matricies for follow up network analysis


def pairwise_spearmanr(data):

    '''pairwse association matrix

    Args:
        data (np.array): N*width

    Returns:
        M (N*N): Asscoation matrix with each element contains a coefficient
        p : p values for each coefficient
    '''
    N, _ = data.shape
    M = np.empty((N, N))
    p = np.empty((N, N))
    dic = {}
    for i in range(N):
        for j in range(N):
            if str(i)+str(j) in dic.keys() or str(j)+str(i) in dic.keys():
                pass
            else:
                corre = spearmanr(data[i, :], data[j, :])
                dic[str(i)+str(j)] = True
                M[i, j] = f'{corre.statistic:.4f}'
                p[i, j] = f'{corre.pvalue:.4f}'
    return M.round(4), p.round(6)

def assocation_series(data, w, s, tstarts, tstop):
    ''' Give pairwise association for sliding time

    Args:
        data (np.array): time series of C for each species, N*time
        w (int): width of the window
        s (int): stride -- how much slide over
        tstarts (int): starting time
        tstop (int): ending time

    Returns:
        series_M (np.array): N*N*num containing association coefficient
        series_p (np.array): N*N*num containing p-values
    '''

    ranges = tstop - tstarts
    num = int(((ranges-w)/w) * (w/s)) # number of associations
    N, _ = data.shape
    series_M = np.empty((N, N, num))
    series_p = np.empty((N, N, num))
    for i in range(num):
        window_data = data[:, i*s:i*s+w]
        M, p = pairwise_spearmanr(window_data)
        np.fill_diagonal(M, 0)
        series_M[:, :, i] = M
        series_p[:, :, i] = p
    
    return series_M, series_p

#### Plot Funcs
def plot_pop_trj(para, t, Ct, ax):
    avgm = para.avgm.reshape(para.avgm.shape[0])

# calulate abundance
    for j in range(len(Ct[1, :])):
        Ct[:, j] = Ct[:, j]/avgm

# if abundance smaller than 1 then the species extincts
    for i in range(len(Ct[:, 1])):
        temp = Ct[i, :]
        temp[temp < 1] = 0
        Ct[i, :] = np.rint(temp)

# plot species biomass trajectories
    np.seterr(divide = 'ignore')
    # plt.figure(figsize=(6, 6), dpi=300)
    for i in range(para.N):
        sns.lineplot(x=np.log10(t), y=np.log10(Ct[i, :]), color='grey', alpha=0.3, ax=ax)
    sns.despine(offset=10, trim=True)
    ax.set_title(f'$M_0 = {np.round(para.B0, 3)}$, $B_0 = {np.round(para.B0, 3)}$, $l = {np.round(para.l_base, 4)}$')
    ax.set_xlabel('$log_{10}(Time)$')
    ax.set_ylabel('$log_{10}(Abundance)$')
    ax.text(0, max(np.log10(Ct).flatten()), f'Remaining Species : {len(Ct[Ct[:, -1]>0,-1])}')

def load_data(path, num_files, type:set = {'C', 'R', 'para', 'id', 't', 'all'}):
    
    '''A function to load data

    Returns:
        _type_: _description_
    '''


    data = []
    
    if type == 'C':
        for i in range(num_files):
            data.append(np.load(path+f'\Ct{i+1}.npy'))
        return data
    
    elif type == 'R':
        for i in range(num_files):
            data.append(np.load(path+f'\Rt{i+1}.npy'))
        return data

    elif type == 'para':
        for i in range(num_files):
            data.append(pkl.load(open(path+f'\para{i+1}.pkl', 'rb')))
            return data

    elif type == 'id':
        for i in range(num_files):
            data.append(np.load(path+f'\id{i+1}.npy'))
        return data

    elif type == 't':
        for i in range(num_files):
            data.append(np.load(path+f'\\t{i+1}.npy'))
        return data

    elif type == 'all':
        for i in range(num_files):

            dic_pair = {}
            dic_pair['t'] = np.load(path+f'\\t{i+1}.npy')
            dic_pair['Rt'] = np.load(path+f'\Rt{i+1}.npy')
            dic_pair['Ct'] = np.load(path+f'\Ct{i+1}.npy')
            dic_pair['id'] = np.load(path+f'\id{i+1}.npy')
            dic_pair['para'] = pkl.load(open(path+f'\para{i+1}.pkl', 'rb'))
            dic_pair['scaling exponents'] = (dic_pair['para'].alpha, dic_pair['para'].beta)
            data.append(dic_pair)

        return data