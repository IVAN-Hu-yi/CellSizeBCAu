    '''This is a file to run multiple simulations using latin cube approaches in parallel threads
    '''

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

## parameter initialisation
N = 50
M = 100
assemblenum = 1
para = Paras(N, M)
assemblies = 1
thread_number = 15
# latin cube sampling
sampler = stats.qmc.LatinHypercube(2, seed=10)
sample = sampler.random(thread_number)
l_bound = [0.1, 0.1] # [para.alpha, para.B0]
u_bound = [1, 5]
sample = stats.qmc.scale(sample, l_bound, u_bound)

def thread_function(name, N, M, para:Paras, assemblenum, tstop, teval, scale=True):

    # for integration
    logging.info("Thread %s: starting", name)
    Rt, Ct, t, para = sim_run(N, M, para, 1, tstop=50000, teval=10000)
    logging.info("Thread %s: finishing", name)