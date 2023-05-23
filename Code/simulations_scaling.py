# This is to investigate the effects of cell size scalings on community structures and varying distribution


import logging
import threading
import time
import numpy as np
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
from math import sqrt
import simulation_func
from scipy import stats
from utilities import plot_pop_trj
import multiprocessing
from concurrent import futures

N = 200
M = 100
para_sep = Paras(N, M)
scaling_exponents = np.linspace(0.5, 1.0, 10)
assemblenum = 1
subcommunity_size=50
num_subcommunity= 10
list_exp = []
tstop = 30000
teval = 10000
tol = 1e-3

betas = [(0.72, x) for x in scaling_exponents]
alphas = [(x, 0.72) for x in scaling_exponents]

p, number = int_preferences(N, M, para_sep.mu_c, assemblenum)
D = int_conversion(M, para_sep.Dbase, para_sep.number, assemblenum, para_sep.sparsity)
l = int_l(M, para_sep.l_base, assemblenum)
rho = int_rho(M, para_sep.rho_base, assemblenum)
vmax = int_vmax(N, M, para_sep.v_max_base, p, number, assemblenum)
m = int_mt(N, para_sep.m_base, assemblenum)
avgm = allocate_avgm(N, para_sep.w, assemblenum, para_sep.ymin, para_sep.ymax)


def sim_fuc(name, N, M, para:Paras, assemblenum, tstop, teval, exp_pair, num_subcommunity, subcommunity_size, tol=1e-3, scale=True):
    
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

        # exp change
        para.alpha, para.beta = exp_pair
        # run and store
        if scale:
            pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.B0, para.M0, para.E0, para.alpha, para.beta, para.gamma, para.R_half, para.avgm)

            odec = lambda t, y:odes_scale_size(t, y, *pars)
            detect_ss_odec = lambda t, y: detect_steady_state(t, y, *pars, tol=tol, N=subcommunity_size)
            detect_ss_odec.terminal = True
            detect_ss_odec.direction = -1
            
            result = solve_ivp(
            odec, t_span=[time[0], time[-1]], y0=y0, t_eval=time, events=detect_ss_odec, vectorized=True)
        
        if not scale:
            pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.R_half)
            result = solve_ivp(odes_not_scaled, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=pars, dense_output=True)
        
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

### Simulations

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    if not os.path.exists('F:\Study\FYP\CellSizeBCAu\Data\\alphas'):
        os.makedirs('F:\Study\FYP\CellSizeBCAu\Data\\alphas')

    for j in range(len(alphas)):

        logging.info(f'New pair started, Number {j+1}')
        Ra, Ca, ta, paraa, idxs = sim_fuc(j, N, M, para_sep, assemblenum, tstop, teval, alphas[j], num_subcommunity, subcommunity_size, tol=tol)

        dir = f'F:\Study\FYP\CellSizeBCAu\Data\\alphas\pair_{j+1}'

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

        logging.info(f'Pair Finished, Number {j+1}')
    
    if not os.path.exists('F:\Study\FYP\CellSizeBCAu\Data\\betas'):
        os.makedirs('F:\Study\FYP\CellSizeBCAu\Data\\betas')

    for j in range(len(alphas)):

        logging.info(f'New pair started, Number {j+1}')
        Ra, Ca, ta, paraa, idxs = sim_fuc(j, N, M, para_sep, assemblenum, tstop, teval, betas[j], num_subcommunity, subcommunity_size, tol=tol)

        dir = f'F:\Study\FYP\CellSizeBCAu\Data\\betas\pair_{j+1}'

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

    logging.info(f'Pair Finished, Number {j+1}')