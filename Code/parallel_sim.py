# This is a file to run multiple simulations using latin cube approaches in parallel threads

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
## parameter initialisation
N = 100
M = 250
assemblenum = 1
para = Paras(N, M)
thread_number = 15
tstop = 50000
teval = 10000

# latin cube sampling
sampler = stats.qmc.LatinHypercube(3)
sample = sampler.random(thread_number)
l_bound = [9, 9, 0.001] # [para.M0, para.B0]
u_bound = [10, 10, 0.01]
sample = stats.qmc.scale(sample, l_bound, u_bound)
fig, axes = plt.subplots(3, 5, figsize=(20, 12), dpi=350)
axes = axes.flatten()

def multiprocess_func(name, N, M, para:Paras, assemblenum, tstop, teval, scale=True):
    
    para = Paras(N, M)
    para.M0, para.B0, para.l_base = sample[name]
    # for integration
    logging.info("Thread %s: starting", name)
    Rt, Ct, t, para = sim_run(N, M, para, assemblenum, tstop=tstop, teval=teval)
    logging.info("Thread %s: finishing", name)
    return Rt, Ct, t, para
    
# Simulation
if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with futures.ThreadPoolExecutor() as executor:

        results = [executor.submit(multiprocess_func, index, N, M, para, assemblenum, tstop, teval) for index in range(thread_number)]
        
    
        for i, f in zip(range(thread_number), futures.as_completed(results)):
            _, C, t, para = f.result()
            plot_pop_trj(para, t, C, axes[i])

    if not os.path.exists('F:\Study\FYP\CellSizeBCAu\Results\Figures'):
        os.makedirs('F:\Study\FYP\CellSizeBCAu\Results\Figures')

    fig.savefig('F:\Study\FYP\CellSizeBCAu\Results\Figures\exploratory.png')
    logging.info("Successful!")