import numpy as np
from parametres import Paras
import size_scaled_func

# Initial Conditions
seed = 100 # set seed to ensure reproducible results in each experiment
# assemblenum: int -> set random state for each individual assembly process

def int_R(M, R, assemblenum): 
    '''
        Resource Content at t0
    '''
    # np.random.seed(seed+assemblenum)
    return np.full((M, 1), R)

def allocate_avgm(N, w, assemblenum, ymin, ymax):
    '''Initialised initial cell mass

    Args:
        N (_type_): _description_
        w (_type_): _description_
        assemblenum (_type_): _description_
        ymin (_type_): _description_
        ymax (_type_): _description_
    
    Returns:
        np.array: average mass array of a community
    '''
    np.random.seed(seed+assemblenum)
    y = np.random.beta(1, w, N).reshape(N, 1)
    return 10 ** (ymin + (ymax-ymin)*y)

def int_C(avgm, density):
    '''
        Biomass Content drawn from beta distribution at t0
    '''
    
    return density * avgm

# define parametres

def int_preferences(N, M, mu_c, assemblenum):
    
    '''Guassian sampling of preferences, assume all are generalists

    Returns:
        np.array: N*M matries
        number: number of preferred resources
    '''

    p = np.zeros((N, M))
    number = int(mu_c*M) if int(mu_c*M) > 0 else 1 # number of preferred resources

    for i in range(N):
        x = np.arange(M)
        np.random.seed(i) # ensure for each experiment, each species same preferences
        np.random.shuffle(x)
        idx = x[0:number] # select favored resoruces
        
        np.random.seed(i)
        values = np.random.normal(1/number, 0.001, number).tolist() # initialisation
        for x, j in zip(idx, range(len(values))):
            p[i, x] = values[j]  # assign values
        p[i, :] = p[i, :]/np.sum(p[i, :]) # normalised to 1

    return (p, number)

def int_conversion(M, Dbase, number, assemblenum, sparse=False):

    '''guassian sampling of conversion around Dbase, only allow sysmetrical matrix

    Args:
        M (int): int
        Dbase (float): Guassian mean
        number (float): proportion of resource can be converted
    Returns:
        np.array: shape input_resource * output resource 
    '''

    # np.random.seed(seed+assemblenum)
    np.random.seed(seed+100)

    if sparse: 
        D = np.random.normal(Dbase, Dbase/(M*10), (M, M)).reshape(M, M) # sample conversion
        # D = D * (1-np.tri(*D.shape, k=-1)) # not allow reversible reactions

    else:
        num = int(M*number) if int(M*number) >= 1 else 1
        D = np.zeros((M, M))
        for i in range(M):
            x = np.arange(M)
            np.random.seed(seed+i*100)
            np.random.shuffle(x)
            idx = x[0:num]
            D[i, idx] = np.random.normal(Dbase, Dbase/10, num) # sample conversion
            # D = D * (1-np.tri(*D.shape, k=-1)) # not allow reversible reactions

    return D/np.sum(D, axis=1)[:, np.newaxis] # row-wise normalisation
    
def int_l(M, l, assemblenum, same=True):
    '''
     return leakage
    '''
    np.random.seed(seed+assemblenum)
    if same: return np.array([l]*M).reshape(M, 1)
    else: return np.random.normal(l, l/10, M).reshape(M, 1)

def int_rho(M, rho, assemblenum):
    '''
    define external resource supply
    '''
    # np.random.seed(seed)
    return np.array([rho]*M).reshape(M, 1)

def int_vmax(N, M, v_max_base, p, number, assemblenum):

    '''initialised maximum uptake of each resource 

    Args:
        v_max_base: mean vmax for uptake
        p: preferences for identify preferred resource
        number: number of preferred resources

    Returns:
        np.array : N by M matrix
    '''

    vmax = np.zeros((N, M)) # for non-favored -- no uptake
    
    for i in range(N):
        temp_p = p[i, :] # identify preferred resource
        idlist = np.where(temp_p[temp_p>=0]) # index where p>0
        vmax[i, idlist] = np.random.normal(v_max_base, 0.05, len(idlist))
    
    return vmax

def int_mt(N, m, assemblenum):
    '''initialise maitainence

    Returns:
        np.array : N*1 matrix
    '''
    # np.random.seed(seed+assemblenum)
    np.random.seed(seed)
    return np.random.normal(m, 0.1, (N, 1)).reshape(N, 1)
