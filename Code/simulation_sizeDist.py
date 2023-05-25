# This is to investigate the effects of cell size scalings on community structures and varying distribution


import logging
import threading
import time
import numpy as np
from scipy.integrate import solve_ivp
from parametres import Paras
from initialisation import *
from models import *
import matplotlib.pyplot as plt
from size_scaled_func import *
import datetime as dt
from simulation_func import *
import pandas as pd
import pickle as pkl
import os
from math import sqrt
import simulation_func
from scipy import stats
import multiprocessing
from concurrent import futures

N = 200
M = 100
para_sep = Paras(N, M)
assemblenum = 1
subcommunity_size=50
num_subcommunity= 10
list_exp = []
tstop = 30000
teval = 10000
tol = 1e-3

w_array = np.linspace(1, 2, 30)

p, number = int_preferences(N, M, para_sep.mu_c, assemblenum)
D = int_conversion(M, para_sep.Dbase, para_sep.number, assemblenum, para_sep.sparsity)
l = int_l(M, para_sep.l_base, assemblenum)
rho = int_rho(M, para_sep.rho_base, assemblenum)
vmax = int_vmax(N, M, para_sep.v_max_base, p, number, assemblenum)
m = int_mt(N, para_sep.m_base, assemblenum)

def sim_fuc(name, N, M, para:Paras, assemblenum, tstop, teval, num_subcommunity, subcommunity_size, avgm):
    
    # for integration
    logging.info("Thread %s: starting", name)

    # data storage
    R_assembles = []
    C_assembles = []
    t_assembles = []
    para_assembles = []
    identifier = []

    start = dt.datetime.now()
    for i in range(num_subcommunity):
        
        para = Paras(subcommunity_size, M)
        ## select corresponding species for subcommunity assemblies
        rng = default_rng(seed=assemblenum+i+8*i)
        idx = rng.choice(range(N), size=subcommunity_size, replace=False)
        identifier.append(idx)
        new_p = p[idx, :]
        new_vmax = vmax[idx, :]
        new_m = m[idx, :]
        new_avgm = avgm[idx, :]

        ## Initialised Initial conditions
        R0 = int_R(M, para.R0, assemblenum)
        C0 = int_C(new_avgm, para.x0)

       # Load parametres
        para.paras(C0, R0, l, rho, new_p, new_vmax, new_m, D, new_avgm)
        time = np.linspace(0, tstop, teval)
        y0 = np.concatenate((R0, C0)).reshape(M+subcommunity_size,) # initial conditions
        # run and store
        pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.B0, para.M0, para.E0, para.alpha, para.beta, para.gamma, para.R_half, para.avgm)

        result = solve_ivp(
        odes_scale_size, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=pars, vectorized=True)
    
        
        Rt = result['y'][0:M]
        Ct = result['y'][M:M+subcommunity_size]
        t = result['t']

        print(f'Subcommunity simulation {i+1} completed runtime:{dt.datetime.now()-start}') if (i+1)%5==0 else None


        R_assembles.append(Rt)
        C_assembles.append(Ct)
        t_assembles.append(t)
        para_assembles.append(para)

    logging.info("Thread %s: finishing", name)
    return R_assembles, C_assembles, t_assembles, para_assembles, identifier


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                datefmt="%H:%M:%S")

if not os.path.exists('F:\Study\FYP\CellSizeBCAu\Data\\SizeD'):
    os.makedirs('F:\Study\FYP\CellSizeBCAu\Data\\SizeD')

for j, w in enumerate(w_array):

    logging.info(f'New pair started, Number {j+1}')
    avgm = allocate_avgm(N, w, j, para_sep.ymin, para_sep.ymax)
    Ra, Ca, ta, paraa, idxs = sim_fuc(j, N, M, para_sep, assemblenum, tstop, teval, num_subcommunity, subcommunity_size, avgm=avgm)

    dir = f'F:\Study\FYP\CellSizeBCAu\Data\\SizeD\D_{j+1}'

    if not os.path.exists(dir):
        os.mkdir(dir)

    for i in range(1, len(Ca)+1):

        with open(dir + f'\Ct{i}.npy', 'wb') as f:
            np.save(f, Ca[i-1])
        f.close()

        with open(dir + f'\Rt{i}.npy', 'wb') as f:
            np.save(f, Ra[i-1])
        f.close()

        with open(dir + f'\\t{i}.npy', 'wb') as f:
            np.save(f, ta[i-1])
        f.close()

        with open(dir + f'\para{i}.pkl', 'wb') as f:
            pkl.dump(paraa[i-1], f)
        f.close()

        with open(dir + f'\Id{i}.npy', 'wb') as f:
            np.save(f, idxs[i-1])
        f.close()

    logging.info(f'Finished, Number {j+1}')
