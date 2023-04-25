import numpy as np

def scale_vin(vin, mass, B0, alpha):

    '''v_in scaling

    Returns:
        _type_: _description_
    '''
    #mass[mass<0] = 0
    # return  vin*(B0*(mass**alpha))
    return np.sign(mass) * vin*(B0*(np.abs(mass)**alpha))


def scale_mt(m, mass, M0, beta):
    '''maintanence scaling

    Args:
        m (_type_): _description_
        mass (_type_): _description_
        M0 (_type_): _description_
        alpha (_type_): _description_

    Returns:
        _type_: _description_
    '''
    #mass[mass<0] = 0
    # return m*(M0*(mass)**(1+alpha))
    return np.sign(mass) * m*(M0*(np.abs(mass)**(beta)))
    
def scale_vout(vout, mass, E0, gamma):

    '''leakge scaled

    Returns:
        _type_: _description_
    '''
    #mass[mass<0] = 0
    # return vout*(E0*(mass**gamma))
    return np.sign(mass) * vout*(E0*(np.abs(mass)**gamma))