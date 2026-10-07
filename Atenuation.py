#%%
import numpy as np
import pandas as pd
#%%
'''
Defino primero las funciones dentro de la atenuación para luego poder 
llamarlas desde el script principal.

Distancia de sobrecarga D
D = √[(Rₜ - d)² cos²θ + d(2Rₜ - d)] - (Rₜ - d) cosθ,

Distancia promedio recorrida por el MCP en la sobrecarga
R = 1/(ε²b) · ln((1 + (b/a)Eᵢ) / (1 + (b/a)Eᶠ))

Atenuación de la señal
𝒫 = exp( - D(cos θ) / R(Eᵢ, ε, Eᶠ) )
'''

def D_overburden(d,cos):
    '''
    Calculates the overburden distance D for a given depth d and zenith angle theta.
    Parameters:
        d: depth in meters (of the experiment)
        cos: cosine of the zenith angle [-1;1]
        R_t: radius of the Earth in meters (not a variable, just a constant)
    Returns:
        D: overburden distance in meters
    '''
    if cos > 1 or cos < -1:
        raise ValueError("cos must be in the range [-1, 1]")
    R_t = 6.378e6 # [m] Radio de la tierra
    D = np.sqrt((R_t-d)**2*((cos)**2)+d*(2*R_t-d))-(R_t-d)*cos
    return D # resultado en METROS

def R_average(eps2, Ei, Ef=None, a=None, b=None):
    '''
    Calculates the average distance R that a mCP travels in the earth 
    Parameters:
        eps: mCP charge fraction (e.g., 1e-3 for a mCP with charge 0.001e)
        Ei: initial energy of the mCP in the surface(in GeV)
        Ef: final energy of the mCP in the surface(in GeV)
        a: parameter for the average distance calculation (default: 0.223 GeV/mwe)
        b: parameter for the average distance calculation (default: 4.64e-4 1/mwe)
    Returns:
        R: average distance in meters using the formula R = (1/eps^2)*(1/b)*log((1+(a/b)*Ei)/(1+(a/b)*Ef))*(1/2.65)
    
    b is in 1/mwe units, so we may need to convert it to 1/m since D is 
    in meters. The paper says that this value was calculated for a "standard
    rock shielding so we consider 1 mwe = (1/2.65) m of rock "
    '''

    if a is None:
        a = 0.223 # [GeV/mwe]
    if b is None:
        b = 4.64e-4 # [1/mwe]
    if Ef is None:
        Ef = 0 # [GeV] energy threshold of the detector
    R = ((1/(eps2))*(1/b))*np.log((1+(b/a)*Ei)/(1+(b/a)*Ef)) # [mwe]
    R = R*1/2.65 # [m]
    return R 

def P_arrival(eps2, Ei, d, cos, Ef=None):
    '''
    Calculates the probability of arrival of a mCP at the detector given its charge fraction, initial and final energy, depth of the experiment, and zenith angle.
    eps: mCP charge fraction
    Ei: initial energy of the mCP in GeV
    Ef: final energy of the mCP in GeV
    d: depth of the experiment in meters
    cos: cosine of the zenith angle [-1;1]
    returns: probability of arrival at the detector
    '''
    if Ef is None:
        Ef = 0 # [GeV] energy threshold of the detector
    D = D_overburden(d, cos) # Calculate overburden distance D
    R = R_average(eps2, Ei, Ef) # Calculate average distance R that the mCP travels in the earth
    P = np.exp(-D/R) # Probability of arrival using the formula P = exp(-D/R)
    return P
#%%
def Atenuation_csv(df,d):
    '''
    Calculate the atenuated flux of mCP at a experiment located at depth d, given a dataframe with the mCP flux at the surface.
    The dataframe must have the following columns: 'eps2', 'm(GeV)', 'cos_theta', 'Energy(GeV)', 'dphi_dEdCth(1/GeV/s/cm2/sr)'.
    The function will return a new dataframe with the same columns, but with the atenuated flux at the experiment. Te atenuation 
    depends on the depth of the experiment, the charge fraction of the mCP, and the zenith angle.
    Parameters:
        df: dataframe with the mCP flux at the surface
        d: depth of the experiment in meters
    Returns:
        df_atenuated: dataframe with the atenuated flux at the experiment
    '''
    if 'dphi_dEdCth(1/GeV/s/cm2/sr)' not in df.columns:
        raise ValueError("The dataframe must have the column 'dphi_dEdCth(1/GeV/s/cm2/sr)'")
    if 'cos_theta' not in df.columns:
        raise ValueError("The dataframe must have the column 'cos_theta'")
    if 'eps2' not in df.columns:
        raise ValueError("The dataframe must have the column 'eps2'")
    if 'Energy(GeV)' not in df.columns:
        raise ValueError("The dataframe must have the column 'Energy(GeV)'")
    cosine_values = df['cos_theta'].unique()
    eps2_values = df['eps2'].unique()
    df_atenuated = df.copy()
    for cos in cosine_values:
        for eps2 in eps2_values:
            mask = (df['cos_theta'] == cos) & (df['eps2'] == eps2)
            Ei_values = df.loc[mask, 'Energy(GeV)'].values
            P_values = np.array([P_arrival(eps2, Ei, d, cos) for Ei in Ei_values])
            df_atenuated.loc[mask, 'dphi_dEdCth(1/GeV/s/cm2/sr)'] *= P_values
    return df_atenuated
#%%
