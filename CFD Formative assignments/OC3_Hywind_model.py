# -*- coding: utf-8 -*-
"""
OC3 Hywind 2D Model
OE44185 - Numerical Methods for Offshore Engineering

"""

# %% IMPORT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar # for wave seed generation
import pandas as pd
import time
import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Directory:", current_directory)


start_time = time.time()

#%% GET THE MOORING DATA
# These tables are generated from the mooring line script.

# The mooring offsets that converged to the specified horizontal forces.
mooring_offset = np.array([0, 3.49,  7.28, 10.15, 12.31, 13.9 , 15.12, 16.23, 17.15, 17.98,
          18.74, 19.5 , 20.22, 20.89, 21.54, 22.19, 22.85, 23.5 , 24.13,
          24.75, 25.38, 25.97, 26.59, 27.22, 27.84, 28.44, 29.05, 29.67,
          30.25, 30.86, 31.48])

mooring_hforce = np.linspace(0, 3e6, 31)

# Fx, Fz components on left, right hand sides
mooring_table = np.array([[  -301446.60422046,   -349797.48464005,   -410631.37652927,
           -470825.79506971,   -556149.09198733,   -645566.18232553,
           -734675.91644729,   -833331.58786685,   -927824.59341565,
          -1022160.27303207,  -1115123.41246619,  -1213381.04277708,
          -1310492.89336252,  -1403732.66190896,  -1496351.12914373,
          -1590714.92691488,  -1687995.05852086,  -1784986.08038156,
          -1879919.1421873 ,  -1974086.30735597,  -2070403.15388061,
          -2161091.876147  ,  -2256823.13777638,  -2354484.90199056,
          -2450921.91636203,  -2544512.82574782,  -2639895.02460998,
          -2737049.1603971 ,  -2828101.84453703,  -2924017.01692269,
          -3021646.51349001],

       [ -2687887.6881004 ,  -3004799.94010243,  -3527369.94912642,
          -4198179.18029163,  -4958975.40782282,  -5756274.47459234,
          -6550832.95985101,  -7430509.03135846,  -8273068.15350561,
          -9114224.45860071,  -9943142.33139713, -10819269.21780724,
         -11685179.61090077, -12516564.08300457, -13342408.64149557,
         -14183815.66575911, -15051226.55836705, -15916059.5664874 ,
         -16762542.50724299, -17602196.23143975, -18461017.86786101,
         -19269655.60541241, -20123255.79827353, -20994069.56745934,
         -21853962.69605019, -22688478.16092576, -23538965.8904145 ,
         -24405253.32498528, -25217136.37572034, -26072376.4326589 ,
         -26942902.48319222],

       [   315260.63321602,    240340.22133834,    210548.57039374,
            185368.97616293,    164951.50323532,    158680.01225456,
            146983.8736424 ,    145386.5577522 ,    140036.26571395,
            131021.36772318,    128771.53864652,    128483.45088051,
            126479.43383188,    122553.74594811,    115161.93944116,
            113569.01718246,    113591.64593374,    112059.31853292,
            108700.09507975,    107329.48739852,    106005.16000889,
             99718.59121803,     98526.42697658,     95561.12345424,
             94435.99487217,     94568.53430948,     93484.57134842,
             92433.88998479,     86963.23910028,     86009.09242459,
             83442.96785209],

       [ -2693198.75156988,  -2396810.98317966,  -2140844.1623609 ,
          -1975535.70580681,  -1865967.23116873,  -1789537.39082197,
          -1736371.80912458,  -1688767.71969355,  -1651371.05794757,
          -1619547.19064498,  -1591737.18969741,  -1562691.06139977,
          -1538317.03106889,  -1514335.98708205,  -1492701.74259443,
          -1472054.66211879,  -1449827.49668397,  -1430269.62883749,
          -1410625.81525124,  -1392839.12816227,  -1375653.03092578,
          -1358563.91305217,  -1342321.89341394,  -1324478.49555432,
          -1308884.19781243,  -1292147.63910165,  -1277336.79116759,
          -1262980.69002487,  -1248574.86145418,  -1234875.69884554,
          -1219926.43058617]])

mooring_table[1] -= mooring_table[1][0]
mooring_table[3] -= mooring_table[3][0]

def Fmoor(x):
    # return np.interp(x, mooring_offset, mooring_hforce)
    fxl = np.interp(x, mooring_offset, mooring_table[0])
    fzl = np.interp(x, mooring_offset, mooring_table[1])
    fxr = np.interp(x, mooring_offset, mooring_table[2])
    fzr = np.interp(x, mooring_offset, mooring_table[3])
    
    return fxl, fzl, fxr, fzr

# Create a figure with 1x3 subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# Scale factor
scale_factor = 1e6

# Plot Horizontal Force in MN
axs[0].plot(mooring_offset, (mooring_table[0] + mooring_table[2]) / scale_factor, label='Horizontal Force', color='blue')
axs[0].set_title('Horizontal Force')
axs[0].set_xlabel('Mooring Offset')
axs[0].set_ylabel('Force [MN]')
axs[0].grid()
axs[0].legend()

# Plot Vertical Force in MN
axs[1].plot(mooring_offset, (mooring_table[1] + mooring_table[3]) / scale_factor, label='Vertical Force', color='green')
axs[1].set_title('Vertical Force')
axs[1].set_xlabel('Mooring Offset')
axs[1].set_ylabel('Force [MN]')
axs[1].grid()
axs[1].legend()

# Plot Moment in MNm
axs[2].plot(mooring_offset, 5.2 * (mooring_table[1] - mooring_table[3]) / scale_factor, label='Moment', color='red')
axs[2].set_title('Moment')
axs[2].set_xlabel('Mooring Offset')
axs[2].set_ylabel('Moment [MNm]')
axs[2].grid()
axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined plot
plt.show()

# Calculate the linearized stiffnesses
fxl, fzl, fxr, fzr = Fmoor(1)
kx_mooring = fxr+fxl
kz_mooring = fzl+fzr

print('Lateral Mooring Stiffness = ',kx_mooring)
print('Vertical Mooring Stiffness =',kz_mooring)

#%% DEFINE THRUST DATA
thrust_table = {
"x":[0, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
"y":[0, 173.913,217.611,284.444,348.282,395.257,479.781,589.497,702.767,807.761,692.916,611.486,543.478,521.739,499.718,456.317,435.26,434.442,413.384,391.587,391.304,390.119,349.407]
}

# Show thrust table
plt.figure()
plt.plot(thrust_table['x'],thrust_table['y'],'o-',color='black')
plt.xlabel('Wind Speed: U [m/s]')
plt.ylabel('Thrust Force: F [kN]')
plt.grid()

#%% WAVE AND WIND METHODS
#Calculates JONSWAP spectrum for waves
def calculateJONSWAPSpectrum(waveDict):
           
    Hs = waveDict["Hs"]
    Tp = waveDict["Tp"]
    
    # If defined, get the gamma. Ohterwise default to 1.0
    gamma = waveDict.get("gamma", 1.0)
                    
    # Calculate frequency information
    df = waveDict["TDur"]**-1.0
    f = np.arange(df, waveDict["fHighCut"] + df, df)
    
    # Spectral width parameter
    sigma = np.ones(len(f))

    fp = 1./Tp

    sigma[f>fp] = 0.09
    sigma[f<=fp] = 0.07        

    # Calculate the Kaimal spectrum
    # Pierson-Moskowitz spectrum
    # Spm = 5/16 * Hs**2 * fp**4 * f**(-5) * \
    #         np.exp( - 5/4 * (f  / fp)**(-4) ) 
    # Jonswap spectrum
    Spectrum = 0.3125 * Hs**2 * Tp * (f / fp)**(-5) * np.exp(-1.25 * (f / fp)**(-4)) * (1 - 0.287 * np.log(gamma)) * gamma**(np.exp(-0.5 * (((f/fp) - 1) / sigma)**2)) 
    amplitudeSpectrum = np.sqrt(2*Spectrum*df)
    
    # Store it inside the wind dictionary
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["Spectrum"] = Spectrum
    outputDict["amplitudeSpectrum"] = amplitudeSpectrum
    outputDict["f"] = f
    
    return outputDict

#Calculates phases to obtain irregular waves
def generateRandomPhases(inputDict, seed=2):
    # Stating a seed will allow for repeatability
    # of your randomness
    #rng = np.random.default_rng(seed)
    #phi = 2*np.pi*rng.random(len(inputDict["Spectrum"]))
    phi = lcg(seed, n=len(inputDict["Spectrum"]))
        
    outputDict = dict()
    outputDict.update(inputDict)    
    outputDict["randomPhases"] = phi;
    
    return outputDict

#Weird function necessary for calculating the seed
def lcg(seed, a=1103515245, c=12345, m=2**31, n=1):
    # Linear congruential random number generator.
    # Chosen to have the sampe implementation in Matlab and Python.
    # See: https://en.wikipedia.org/wiki/Linear_congruential_generator
    
    numbers = np.ones(n)
    numbers *= seed
    for i in range(1, n):
        numbers[i] = (a * numbers[i-1] + c) % m
    return 2*np.pi*np.array([x / m for x in numbers])  # Normalize to [0, 1]

#Function for dispersion relation called by the kinematics function
def dispersion(f, h):
    N = len(f)
    omega = 2*np.pi*f
    k = np.zeros_like(f)
    g = loadConstants["g"]

    myFun = lambda k, omega, g, h: omega**2 - g*k*np.tanh(k*h)
    myFunPrime = lambda k, omega, g, h: -g*(k*h*(1-np.tanh(k*h)**2)+np.tanh(k*h));

    kGuess = omega[0] / np.sqrt(g*h)

    for j in range(N):
        k[j] = root_scalar(lambda x: myFun(x,omega[j],g,h), 
                                fprime=lambda x: myFunPrime(x,omega[j], g,h), x0=kGuess, 
                                method='newton').root;
        kGuess = k[j]
        
    return k


#fft variant of calculating free surface elevation
def calculateFreeSurfaceElevationTimeSeriesFFT(waveDict):
    
    t = waveDict["t"]
    # f = waveDict["f"]
    
    M = len(t)
    freeSurfTimeSeriesKernel = waveDict["amplitudeSpectrum"] * np.exp(1j * waveDict["randomPhases"])
    freeSurfTimeSeriesKernel = pad2(freeSurfTimeSeriesKernel, M) # compute the freq. domain kernel and pad to M
    freeSurfTimeSeries = np.fft.ifft(freeSurfTimeSeriesKernel).real*M # perform the IFFT in this line
    
    # Store the result
    outputDict = dict()
    outputDict.update(waveDict)    
    outputDict["t"] = t
    outputDict["eta"] = freeSurfTimeSeries
        
    return outputDict

#Calculate kinematics for free surface elevation
def calculateKinematicsFFT(inputDict):
    
    t = inputDict["t"]
    f = inputDict["f"]
    omega = 2*np.pi*f
    
    h = inputDict["h"]
    z = inputDict["z"]
    u = np.zeros((len(t), len(z)))
    ut = np.zeros((len(t), len(z)))
    phi = np.zeros((len(t), len(z)))
    
    k = dispersion(f, inputDict["h"])
    M = len(t)
    
    for j_, z_ in enumerate(z):
        
        uKernel = inputDict["amplitudeSpectrum"] * omega * np.cosh(k*(z_ + h))/ np.sinh(k * h)*np.exp(1j*inputDict["randomPhases"])  # compute the freq. domain kernel and pad to M
        u[:, j_] = np.fft.ifft(pad2(uKernel, M)).real*M # perform the IFFT in this line
        
        utKernel = inputDict["amplitudeSpectrum"] * omega**2 * np.cosh(k*(z_ + h))/ np.sinh(k * h)*np.exp(1j*(inputDict["randomPhases"] + (np.pi/2))) # compute the freq. domain kernel and pad to M
        ut[:, j_] = np.fft.ifft(pad2(utKernel, M)).real*M # perform the IFFT in this line
        
        pKernel = -loadConstants["rho_water"] * inputDict["amplitudeSpectrum"] * loadConstants["g"] * np.cosh(k*(z_ + h))/ np.sinh(k * h)*np.exp(1j*inputDict["randomPhases"])
        phi[:, j_] = np.fft.ifft(pad2(pKernel, M)).real*M
    
    outputDict = dict()
    outputDict.update(inputDict)        
    outputDict["u"] = u
    outputDict["ut"] = ut
    outputDict["phi"] = phi
    
    return outputDict

#Necessary for the fast forward fourier transformation
def pad2(vector, size):
    # Padding an array with zeros up to size
    return np.pad(vector, [1, size - len(vector) - 1])

#Function to calculate random wind spectrum
def calculateKaimalSpectrum(windDict):
    
    # Store it inside the wind dictionary
    outputDict = dict()
    outputDict.update(windDict)
    
    # Calculate frequency information
    df = windDict["TDur"]**-1
    f = np.arange(df, windDict["fHighCut"], df)
    #print("Length of f:", len(f))

    # Calculate the Kaimal spectrum
    Spectrum = (4 * windDict["I"]**2 * windDict["V_10"] * windDict["l"]) / (1 + 6 * ((f * windDict["l"])/windDict["V_10"]))**(5/3)
    print(windDict["I"])
    amplitudeSpectrum = np.sqrt(2*Spectrum*df)

    outputDict["Spectrum"] = Spectrum
    outputDict["amplitudeSpectrum"] = amplitudeSpectrum
    outputDict["f"] = f
    
    return outputDict

#Generates wind time series by stacking different random frequencies of the Kaimal spectrum on top of each other
def calculateWindTimeSeriesFFT(windDict):
    t = windDict["t"]
    # f = windDict["f"]
    windTimeSeries = np.zeros_like(t)
    
    M = len(t)
    windTimeSeriesKernel = windDict["amplitudeSpectrum"] * np.exp(1j * windDict["randomPhases"]) # compute the freq. domain kernel and pad to M
    windTimeSeriesKernel = pad2(windTimeSeriesKernel, M)
    windTimeSeries = np.fft.ifft(windTimeSeriesKernel).real * M # perform the IFFT in this line
    
    # Store the result
    outputDict = dict()
    outputDict.update(windDict)
    outputDict["t"] = t
    outputDict["V_hub"] = windTimeSeries + windDict["V_10"]
    
    return outputDict

#%% WIND AND WAVE VARIABLE DICTIONARIES
#Variable dictionaries
timeInfo = {"TTrans":50,
            "TDur":501,
            "dt":50/100,
            "fHighCut":0.5
           }

loadConstants = {
    "g": 9.81,
    "rho_air": 1.22,
    "rho_water": 1025.0
}

wind = {
    "I": 0.14,
    "l": 340.2,
    "V_10": 8.0
}

wave = {
    "Hs" : 7.5,
    "Tp" : 11,
    "gamma" : 1.0,
    "h" : 320,
    "z" : np.arange(-320, 1, 1)
}

#%% GENERATE IRREGULAR WAVES
#Generating irregular waves
seed = 2

print(timeInfo)

wave.update(timeInfo)
waves = calculateJONSWAPSpectrum(wave)
waves = generateRandomPhases(waves, seed=2)
waves["t"] = np.arange(0., waves["TDur"], waves["dt"])
waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
waves = calculateKinematicsFFT(waves)

#plot the first 1000 entries of the wave
plt.figure(figsize=(20, 10))
plt.title("Irregular wave data over 1200 seconds")
plt.plot(waves["t"],waves["eta"]);

#%% GENERATE THE WIND SPECTRUM
#Generate the wind spectrum
seed = 2

wind.update(timeInfo)
wind["t"] = np.arange(0., wind["TDur"], wind["dt"])
#print(wind)
wind = calculateKaimalSpectrum(wind)
#print(wind["amplitudeSpectrum"])
wind = generateRandomPhases(wind)
wind = calculateWindTimeSeriesFFT(wind)

plt.figure(figsize=(10, 5))
plt.title("Wind speeds")
plt.plot(wind["t"],wind["V_hub"])
plt.grid()

#%% MATERIAL PROPERTIES

# Elastic modulus
E = 210e9 # [Pa]

# Shear modulus
G = 80.8e9 # [Pa]

# Effective density of steel
rho = 8500 # [kg/m^3]

rho_water = 1025 # [kg/m^3]


#%% DEFINE THE CURRENT FORCE
# Taken from the CFD analysis
Fcurrent_avg = 1797

print('Average current force = ',Fcurrent_avg)


#%% TOWER PROPERTIES
# Tower properties taken from Phase IV of OC3 document
data = {
    "Elevation": [10.00, 17.76, 25.52, 33.28, 41.04, 48.80, 56.56, 64.32, 72.08, 79.84, 87.60],
    "HtFract": [0.00000, 0.10000, 0.20000, 0.30000, 0.40000, 0.50000, 0.60000, 0.70000, 0.80000, 0.90000, 1.00000],
    "TMassDen": [4667.00, 4345.28, 4034.76, 3735.44, 3447.32, 3170.40, 2904.69, 2650.18, 2406.88, 2174.77, 1953.87],
    "TwFAStif": [603.903e+9, 517.644e+9, 440.925e+9, 373.022e+9, 313.236e+9, 260.897e+9, 215.365e+9, 176.028e+9, 142.301e+9, 113.630e+9, 89.488e+9],
    "TwSSStif": [603.903e+9, 517.644e+9, 440.925e+9, 373.022e+9, 313.236e+9, 260.897e+9, 215.365e+9, 176.028e+9, 142.301e+9, 113.630e+9, 89.488e+9],
    "TwGJStif": [464.718e+9, 398.339e+9, 339.303e+9, 287.049e+9, 241.043e+9, 200.767e+9, 165.729e+9, 135.458e+9, 109.504e+9, 87.441e+9, 68.863e+9],
    "TwEAStif": [115.302e+9, 107.354e+9, 99.682e+9, 92.287e+9, 85.169e+9, 78.328e+9, 71.763e+9, 65.475e+9, 59.464e+9, 53.730e+9, 48.272e+9],
    "TwFAIner": [24443.7, 20952.2, 17847.0, 15098.5, 12678.6, 10560.1, 8717.2, 7124.9, 5759.8, 4599.3, 3622.1],
    "TwSSIner": [24443.7, 20952.2, 17847.0, 15098.5, 12678.6, 10560.1, 8717.2, 7124.9, 5759.8, 4599.3, 3622.1],
    "TwFAcgOf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "TwSScgOf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Create the DataFrame
tower = pd.DataFrame(data)

# Display the DataFrame
print(tower)

# Tower elevation
elev_tower = tower.Elevation # [m]

# Mass density
rA = tower.TMassDen # [kg/m]

# Forward-aft bending stiffness = side-side bending stiffness = EI
EI = tower.TwFAStif # [N-m^2]

# Torsional stiffness
GJ = tower.TwGJStif # [N-m^2]

# Axial stiffness
EA = tower.TwEAStif # [N]

# Bending inertia 
rI = tower.TwFAIner # [kg-m]

# Assumption radius and thickness of tower are assumed to be linearly tapered
# Base diameter
D_base = 6.5 # [m]

# Base thickness
t_base = 0.027 # [m]

# Top diameter
D_top = 3.87 # [m]

# Top thickness
t_top = 0.019 # [m]

# Discretization: number of elements
nel_tower = 15

# Number of nodes
nn_tower = nel_tower + 1

# Create the diameter and thickness ranges
D_tower = np.linspace(D_base, D_top, nel_tower) # m

# Height of the tower 
z_top = 87.6 # m above SWL
z_base = 10 # m above SWL

# Nodal positions
zn_tower = np.linspace(z_base, z_top, nn_tower) # [m]

# Elevation map
z_tower = np.linspace(z_base, z_top, nel_tower) # [m]

# Properties
rA_tower = np.interp(z_tower, elev_tower, rA)
EI_tower = np.interp(z_tower, elev_tower, EI)
EA_tower = np.interp(z_tower, elev_tower, EA)
rI_tower = np.interp(z_tower, elev_tower, rI)

rho_water = 1025
rAw_tower = rho_water*np.pi*(D_tower/2)**2

# First iteration:
i = 0

# Interpolate betwewn the points
R_tower = np.interp(z_tower, [z_base, z_top], [D_base, D_top])/2
t_tower = np.interp(z_tower, [z_base, z_top], [t_base, t_top])

# Plot the tower outer surface
plt.figure()
plt.plot(R_tower, z_tower)
plt.plot(-R_tower, z_tower)
plt.grid()

# Plot the thickness over the height of the tower
plt.figure()
plt.plot(t_tower, z_tower)
plt.grid()

# Height of the hub above SWL, undeflected
H_hub = 90 # [m]

# Distance from tower top to hub height
dz_hub = 2.4 # [m] 

# Shaft tilt
tilt_hub = 5 # [deg]

# Hub center to yaw axis
hub_center_to_yaw_axi = 5.0191 # [m]

# Vertical distance from yaw axis to tower top 
dz_yaw_axis = 1.96256 # [m]

# Hub mass like in the REpower 5 M
m_hub = 56780 # [kg]

# Hub position
u_offset = 5 # [m] upwind of the tower centerline

# Center of Gravity position
cog_hub = np.array([u_offset, 0, H_hub]) # [m]

# Hub inertia about the shaft
I_hub = 115926 # [kg-m^2]

# Hub casting
r_hub = 1.75 # [m] spherical radius

#%% BLADE PROPERTIES

# Blade mass (overall, integrated)
m_blade = 17740 # [kg] 

# Second mass moment of inertia w.r.t root
I_blade = 11776047 # [kg-m^2]

cog_blade = np.array([u_offset, 0, H_hub])

#%% NACELLE PROPERTIES
# Nacelle mass like in the REpower 5M
m_nacelle = 240000 # [kg]


# Height of the nacelle
H_nacelle = 90

# Center of gravity of the nacelle
cog_nacelle = np.array([-1.9, 0, H_nacelle]) # [m] downwind of yaw axis

# Nacelle inertia about the yaw axis
# Translated using the parallel axis theorem with the nacelle mass and downwind
# distance to the nacelle CM
I_nacelle = 2607890 # [kg-m^2] equivalent to DOWC turbine nacelle inertia

#%% ROTOR-NACELLE ASSEMBLY
# Create the hub and nacelle node
m_RNA = m_hub + m_nacelle + 3*m_blade

# Center of gravity of the rotor-nacelle assembly
cog_RNA = (m_hub*cog_hub + m_nacelle*cog_nacelle + 3*m_blade*cog_blade) \
     / (m_hub + m_nacelle + 3*m_blade)
     
# Second moment of inertia about axis Z5 (pitch)
I55_RNA = m_RNA*(cog_RNA[0]**2+(H_hub - z_top)**2) + 2*I_blade
    
# Define the mass matrix of the RNA:
M_RNA = np.array([[m_RNA, 0, 0],
                  [0, m_RNA, 0],
                  [0, 0, I55_RNA]])
   
plt.figure()
plt.spy(M_RNA)

#%% OC3-HYWIND PLATFORM PROPERTIES

# Draft
draft = -120 # [m] below SWL

z_base_buoy = draft # [m] below SWL

# Elevation to platform top
z_top_buoy = 10 # [m] above SWL

# Depth to top of taper
z_top_taper_buoy = -4 # [m] below SWL

# Depth to bottom of taper below SWL
z_base_taper_buoy = -12 # [m] below SWL

# Height of taper
H_taper_buoy = np.abs(z_top_taper_buoy - z_base_taper_buoy)

# Diameter at and above taper
D_above_buoy = 6.5 # [m]

# Diameter below taper
D_below_buoy = 9.4 # [m]

# Change in diameter
D_delta_buoy = D_below_buoy - D_above_buoy

# Slope of taper
taper_slope_buoy = D_delta_buoy / H_taper_buoy # [m/m]

# Mass, including ballast, of the floating platform
m_buoy = 7466330 # [kg] 

# Center of gravity below still water line
cog_buoy = -89.9155 # [m]

# Roll inertia about CM
I44_buoy = 4229230000 # [kg-m^2]

# Pitch inertia about CM
I55_buoy = 4229230000 # [kg-m^2]

# Yaw inertia about CM
I66_buoy = 164230000

#%% BUOY PROPERTIES
zn_range = np.array([z_base_buoy,  z_base_taper_buoy, z_top_taper_buoy, 0, z_top_buoy])
R_buoy = np.array([D_below_buoy/2, D_below_buoy/2, D_above_buoy/2, D_above_buoy/2, D_above_buoy/2])
RO_buoy = R_buoy
t_buoy = np.ones_like(zn_range)*0.0365 # [m]

A_buoy = np.pi*(R_buoy**2 - (R_buoy-t_buoy)**2)
# z_buoy_interp = np.interp(np.linspace(-120,0,100),z_buoy,z_buoy)
V0_buoy = np.trapz(np.pi*R_buoy**2,zn_range) - np.pi*R_buoy[-1]**2*z_top_buoy

# Gravity
g = 9.81

# Buoyancy force
Fb = rho_water*V0_buoy*g

def Fb_fn(zeta):
    Fb = 0
    return Fb

# Plot the interpolated external buoy area to verify
plt.figure()
plt.plot(zn_range, A_buoy,'b-')
# plt.plot(z_buoy, A_buoy,'k.')

# Find the center of mass of the buoy without ballast
# z_buoy_interp = np.interp(np.linspace(z_buoy[0],z_buoy[-1],100),z_buoy,z_buoy)
# rA_buoy_interp = np.interp(np.linspace(z_buoy[0],z_buoy[-1],100),z_buoy,rA_buoy)
rA_buoy = rho*A_buoy
CM_shell = np.trapz(rA_buoy*zn_range,zn_range)/np.trapz(rA_buoy,zn_range)

print('Buoy Shell Center of Mass = ',CM_shell,'m')

# Then find the center of mass of the ballast
# m_shell = np.trapz(rA_buoy_interp,z_buoy_interp)
m_shell = np.trapz(rA_buoy, zn_range)
m_total = m_buoy - m_RNA
m_ballast = m_total - m_shell

print('Total mass = ',m_total,'kg')
print('Shell mass = ',m_shell,'kg')
print('Ballast Mass =',m_ballast,'kg')

CM_ballast = (cog_buoy*m_total - m_shell*CM_shell)/m_ballast

print('Ballast Center of Mass = ',CM_ballast)

# Verify the CM is in the correct location
cog_buoy_check = (CM_ballast*m_ballast + CM_shell*m_shell)/(m_shell+m_ballast)

print('Buoy Center of Mass (Check) = ',cog_buoy_check,'m')

# Buoy center of buoyancy
cob_buoy_check = np.trapz(np.pi*RO_buoy[0:3]**2*zn_range[0:3], zn_range[0:3]) \
    /np.trapz(np.pi*RO_buoy[0:3]**2, zn_range[0:3])

print('Center of Buoyancy CB = ',cob_buoy_check)

# Determine the volume of the ballast assuming it extends to the base of the buoy
H_ballast = 2*(CM_ballast - zn_range[0])
print('Height of ballast = ',H_ballast,'m')


# Ballast Volume = height x internal area of the buoy
V_ballast = H_ballast*np.pi*(R_buoy[0]-t_buoy[0])**2

# Ballast density = mass / volume
rho_ballast = m_ballast / V_ballast

print('Ballast density = ',rho_ballast,'kg/m^3')

# Define the ballast density times area
# rA_ballast = ballast density x internal area
rA_ballast = rho_ballast*np.pi*(R_buoy-t_buoy)**2

# Plot the radius on both sides of the centerline
plt.figure()
plt.plot(R_buoy, zn_range,color='black',linewidth=3,label='Border')
plt.plot(-R_buoy, zn_range,color='black',linewidth=3)
plt.plot(np.linspace(-R_buoy[0],R_buoy[0],2),
         np.linspace(zn_range[0],zn_range[0],2),color='black',linewidth=3)
plt.grid()

# Plot the center of gravity of the buoy
plt.plot(0, cog_buoy,color='black',marker='o',markersize=12,
         label='Center of Gravity')

# Superimpose the tower
plt.plot(R_tower, z_tower)
plt.plot(-R_tower, z_tower)

# Include the tower nodes
plt.scatter(np.zeros(nn_tower), zn_tower)

# Include the buoy nodes
plt.scatter(np.zeros(len(zn_range)), zn_range)

# Create a legend
plt.legend()

plt.show()

#%% DISCRETIZE BUOY
# Number of platform elements
nel_buoy = 15

# Number of buoy nodes
nn_buoy = nel_buoy + 1

zn_range = [z_base_buoy, draft+H_ballast, z_base_taper_buoy, \
            z_top_taper_buoy, 0, z_top_buoy]  # Specified key node positions

def interpolate_nodes_preserving_keys(zn_range, total_nodes):
    zn_range = np.array(zn_range)
    
    # Total number of segments needed
    total_segments = total_nodes - 1

    # Length of each segment between key nodes
    seg_lengths = np.diff(zn_range)
    total_length = np.sum(seg_lengths)

    # Proportional number of segments in each interval
    raw_segments = seg_lengths / total_length * total_segments
    seg_counts = np.round(raw_segments).astype(int)

    # Ensure exact number of segments is total_segments
    while np.sum(seg_counts) != total_segments:
        diff = total_segments - np.sum(seg_counts)
        idx = np.argmax(raw_segments - seg_counts if diff > 0 else seg_counts - raw_segments)
        seg_counts[idx] += np.sign(diff)

    # Generate nodes
    zn = []
    for i in range(len(zn_range) - 1):
        n_points = seg_counts[i] + 1  # include left endpoint
        nodes = np.linspace(zn_range[i], zn_range[i+1], n_points, endpoint=False)
        zn.extend(nodes)

    zn.append(zn_range[-1])  # include last key node
    return np.array(zn)

zn_buoy = interpolate_nodes_preserving_keys(zn_range, nn_buoy)
print("Generated nodes:", zn_buoy)

nel_buoy = len(zn_buoy) - 1

# Print the node positions
print("Node positions:", zn_buoy)

# # Create the radius profile
R_buoy_interp = np.zeros(nel_buoy)
rA_buoy_interp = np.zeros(nel_buoy)
rI_buoy_interp = np.zeros(nel_buoy)

# # For every point along the buoy
i = 0
for z in (zn_buoy[:-1] + zn_buoy[1:])/2:
# for z in zn_buoy[1:]:
    
    # If using ballast:
    if z < draft+H_ballast:
        rA_buoy_interp[i] = rA_buoy[0] + rA_ballast[0]
        R_buoy_interp[i] = R_buoy[0]
        rI_buoy_interp[i] = rho*np.pi/4*(RO_buoy[0]**4 - (RO_buoy[0]-t_buoy[0])**4) \
            + np.pi/4*rho_ballast*(RO_buoy[0] - t_buoy[0])**4
        
    elif z <= z_base_taper_buoy:
        rA_buoy_interp[i] = rA_buoy[1]
        R_buoy_interp[i] = R_buoy[1]
        rI_buoy_interp[i] = rho*np.pi/4*(RO_buoy[1]**4 - (RO_buoy[1]-t_buoy[0])**4)
        
    elif z <= z_top_taper_buoy:
        R_buoy_interp[i] = (D_below_buoy
                      - taper_slope_buoy * (z - z_base_taper_buoy)) / 2
        
        rA_buoy_interp[i] = np.interp(z,zn_range[1:],rA_buoy)
        
        rI_buoy_interp[i] = rho*np.pi/4*(R_buoy_interp[i]**4 - (R_buoy_interp[i]-t_buoy[0])**4)
        # rA_buoy_interp[i] = rA_buoy[1] + (rA_buoy[2] - rA_buoy[1])/(z_top_taper_buoy - z_base_taper_buoy) * (z - z_base_taper_buoy)
        
    else:
        R_buoy_interp[i] = D_above_buoy / 2
        rA_buoy_interp[i] = rA_buoy[2]
        rI_buoy_interp[i] = rho*np.pi/4*(RO_buoy[-1]**4 - (RO_buoy[-1]-t_buoy[0])**4)
        
    i += 1
    
R_buoy = R_buoy_interp
rA_buoy = rA_buoy_interp
rI_buoy = rI_buoy_interp

# Define a uniform thickness
t_buoy = np.ones(nel_buoy)*0.0365 # [m]

# Define the properties
A_buoy = np.pi*(R_buoy**2 - (R_buoy - t_buoy)**2)
I_buoy = np.pi/4*(R_buoy**4 - (R_buoy - t_buoy)**4)
EA_buoy = E*A_buoy
EI_buoy = E*I_buoy

rho_water = 1025
rAw_buoy = rho_water*np.pi*R_buoy**2

#%% STEP 1: DISCRETIZE THE DOMAIN
# Total number of elements in the beam
ne = nel_buoy + nel_tower

# Total number of nodes in the beam (minus one)
nn = ne + 1

# Collect all of the nodes
# zn = np.append(zn_buoy[:-1], zn_tower)
# zn = np.append(zn_buoy[:-1], zn_tower)
if np.isclose(zn_buoy[-1], zn_tower[0]):
    zn = np.append(zn_buoy[:-1], zn_tower)
else:
    zn = np.append(zn_buoy, zn_tower)

# Get the index of the fairleads
Fair_idx = np.argmin(np.abs(zn - (-70)))
# PLot the nodes
nodes = (zn, np.ones(nn))
plt.figure(figsize=(10, 1))
plt.plot(nodes[0], nodes[1], 'o-')
plt.xlabel('z-coordinate, 0 = SWL');
plt.ylabel('Nodal value')
plt.grid()

# Element-node connectivity
elem_nodes = []
for ie in np.arange(0,ne):
    elem_nodes.append([ie, ie+1])
    
elem_dofs = []
dof_node = []
for ie in np.arange(0, ne):
    elem_dofs.append(np.arange(3 * ie, 3 * ie + 6))
for idof in np.arange(0, 3*nn):
    dof_node.append(int(np.floor(idof / 3)))

#%% COMBINE BUOY AND TOWER PROPERTIES

# Combine the buoy and the tower
rA = np.append(rA_buoy, rA_tower)
rI = np.append(rI_buoy, rI_tower)
rAw = np.append(rAw_buoy, rAw_tower)
EI = np.append(EI_buoy, EI_tower)
EA = np.append(EA_buoy, EA_tower)
Rs = np.append(R_buoy, R_tower)

el_coords = (zn[:-1] + zn[1:])/2

# Plot the results for rho*A
plt.figure()
plt.scatter(el_coords, rA)
plt.xlabel('Nodal z-coordinate [m]')
plt.ylabel('Mass: rA')
plt.grid()

# Plot the axial stiffness
plt.figure()
plt.scatter(el_coords, EA)
plt.xlabel('Nodal z-coordinate [m]')
plt.ylabel('Axial Stiffness: EA')
plt.grid()

# Plot the bending stiffness
plt.figure()
plt.scatter(el_coords, EI)
plt.xlabel('Nodal z-coordinate [m]')
plt.ylabel('Bending Stiffness: EI')
plt.grid()

#%% STEP 2: DEFINE ROD SHAPE FUNCTIONS

# Define the element sizes
h = np.diff(zn)  # [m] height of each element
N_rod = []
dN_rod = []
i = 0

for ie in np.arange(0, ne):
    nodes = elem_nodes[ie]
    print(nodes)
    ze = zn[nodes]
    h_local = h[ie]
    N_rod.append([
        lambda z, ze=ze, h=h_local: (ze[1] - z) / h,
        lambda z, ze=ze, h=h_local: (z - ze[0]) / h
    ])
    dN_rod.append([
        lambda z, h=h_local: -1 / h + 0.0 * z,
        lambda z, h=h_local: 1 / h + 0.0 * z
    ])
    # N_rod.append([lambda z: (ze[1] - z) / h[i], lambda z: (z - ze[0]) / h[i]])  # Changed N to N_rod
    # dN_rod.append([lambda z: -1 / h[i] + 0.0 * z, lambda z: 1 / h[i] + 0.0 * z])  # Changed dN to dN_rod
    i += 1

# Plot the shape functions
plt.figure()
color = iter(plt.cm.rainbow(np.linspace(0, 1, ne + 1)))
c = next(color)
for ie in np.arange(0, ne):
    nodes = elem_nodes[ie]
    ze = zn[nodes]
    # Manage color scheme
    prev = c
    c = next(color)
    p = [prev, c]
    for i in np.arange(0, len(nodes)):
        plt.plot(ze, N_rod[ie][i](ze), c=p[i])  # Changed N to N_rod
plt.title("Shape functions")

plt.figure()
color = iter(plt.cm.rainbow(np.linspace(0, 1, ne + 1)))
c = next(color)
for ie in np.arange(0, ne):
    nodes = elem_nodes[ie]
    ze = zn[nodes]
    # Manage color scheme
    prev = c
    c = next(color)
    p = [prev, c]
    for i in np.arange(0, len(nodes)):
        plt.plot(ze, dN_rod[ie][i](ze), c=p[i])  # Changed dN to dN_rod
plt.title("Shape function derivatives")

#%% STEP 2: DEFINE BEAM SHAPE FUNCTIONS
def base(z, h_loc):
    if isinstance(z, float):
        return np.array([[1], [z], [z**2], [z**3]])
    else:
        return np.array([[np.ones(len(z))], [z], [z**2], [z**3]])

def dbase(z, h_loc):
    if isinstance(z, float):
        return np.array([[0], [1], [2 * z], [3 * z**2]])
    else:
        return np.array([[np.zeros(len(z))], [np.ones(len(z))], [2 * z], [3 * z**2]])

def ddbase(z, h_loc):
    if isinstance(z, float):
        return np.array([[0], [0], [2], [6 * z]])
    else:
        return np.array([[np.zeros(len(z))], [np.zeros(len(z))], [2*np.ones(len(z))], [6 * z]])

def make_N(coeff, h_loc):
    return lambda z: np.dot(np.transpose(base(z, h_loc)), coeff)

def make_dN(coeff, h_loc):
    return lambda z: np.dot(np.transpose(dbase(z, h_loc)), coeff)

def make_ddN(coeff, h_loc):
    return lambda z: np.dot(np.transpose(ddbase(z, h_loc)), coeff)

N_k_b = []
dN_k_b = []
ddN_k_b = []

for ie in np.arange(0, ne):
    h_loc = zn[ie + 1] - zn[ie]  # Length of the current element
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], \
                       [1, h_loc, h_loc**2, h_loc**3], [0, 1, 2 * h_loc, 3 * h_loc**2]])

    dof_vec = np.arange(0, 4)
    N_k_element = []
    dN_k_element = []
    ddN_k_element = []

    for idof in dof_vec:
        rhs = np.zeros(len(dof_vec))
        rhs[idof] = 1
        coeff = np.linalg.solve(matrix, rhs)
        N_k_element.append(make_N(coeff, h_loc))
        dN_k_element.append(make_dN(coeff, h_loc))
        ddN_k_element.append(make_ddN(coeff, h_loc))

    N_k_b.append(N_k_element)
    dN_k_b.append(dN_k_element)
    ddN_k_b.append(ddN_k_element)
    
h = zn[1] - zn[0]  # Use the first element's length for plotting
# zplot = np.arange(0, h + h/100, h/100)

# Plot shape functions and their derivatives
zplot = np.linspace(0, h, 100)
fig, axs = plt.subplots(4, 3, figsize=(12, 10))
for i in range(4):
    N_values = N_k_b[0][i](zplot).T.tolist()
    dN_values = dN_k_b[0][i](zplot).T.tolist()
    ddN_values = ddN_k_b[0][i](zplot).T.tolist()
    axs[i, 0].plot(zplot, N_values[0][:])
    axs[i, 0].set_title(f"Shape function N${i+1}$")
    axs[i, 1].plot(zplot, dN_values[0][:])
    axs[i, 1].set_title(f"Shape function derivative N${i+1}$'")
    axs[i, 2].plot(zplot, ddN_values[0][:])
    axs[i, 2].set_title(f"Shape function derivative N${i+1}$'")

plt.tight_layout()
plt.show()

print(h)

# %% COMPUTATION OF ELEMENT MATRICES: ROD

# Cm for morison equation
Ca = 0.969954
Cd = 0.6

import scipy.integrate as scp
# COMPUTATION OF ELEMENT MATRICES: ROD
M_k_rod = np.zeros((ne, 6, 6))
K_k_rod = np.zeros((ne, 6, 6))

M_k_beam = np.zeros((ne, 6, 6))
K_k_beam = np.zeros((ne, 6, 6))
K_k_coupled = np.zeros((ne, 6, 6))
F_k_beam = np.zeros((ne, 6, 6))  # Force due to gravity
R_k_beam = np.zeros((ne, 6, 6))  # Restoring coefficients
D_k_beam = np.zeros((ne, 6, 6))  # Damping coefficients
I_k_beam = np.zeros((ne, 6, 6))  # Added mass matrix

beam_dofs = {0: 0, 2: 1, 3: 2, 5: 3}
rod_dofs = {1: 0, 4: 1}

beam_dofs = {0: 0, 2: 1, 3: 2, 5: 3}
rod_dofs = {1: 0, 4: 1}

# Include the current loading
def make_Q(eqn, ze):
    return scp.quad(eqn, ze[0], ze[1])[0]

F_current = Fcurrent_avg # [N] current force per meter
Q2cur_k = np.zeros((ne, 6))


T_self = np.zeros(len(zn))
elem_weight = np.zeros(ne)

# Compute weight per element
for ie in range(0,ne):
    n1, n2 = elem_nodes[ie]
    z1, z2 = zn[n1], zn[n2]
    L = abs(z2 - z1)
    elem_weight[ie] = rA[ie] * L


# Accumulate load from top to bottom
for i, z_node in enumerate(zn):
    for ie in range(ne):
        n1, n2 = elem_nodes[ie]
        z_elem = max(zn[n1], zn[n2])  # top of the element
        
        if max(n1, n2) == max(max(elem_nodes)):
                T_self[i] += m_RNA
        
        if z_elem > z_node:  # include the element directly below
            T_self[i] += elem_weight[ie]
            

T_self *= g
print(T_self)

total_weight = sum(rA[i] * abs(zn[elem_nodes[i][1]] - zn[elem_nodes[i][0]]) for i in range(ne))
print(f"Total structural weight: {total_weight:.2e}")

#%% ELEMENT MATRICES
for ie in range(ne):
    nodes = elem_nodes[ie]
    ze = zn[nodes]
    for idof in np.arange(6):

        for jdof in np.arange(6):
            if idof in beam_dofs and jdof in beam_dofs:
                beam_i = beam_dofs[idof]
                def eqn(z): 
                    return F_current*N_k_b[ie][beam_i](z)
                def make_Qcur():
                    return lambda t: [make_Qcur(eqn, ze)]
                #Q_k.append([make_Q(eqn1, xe), make_Q(eqn2, xe)])
                # Qcur_k.append(make_Q(eqn, ze))
                if zn[ie + 1] <= 0:
                    Q2cur_k[ie, idof] = scp.quad(eqn, 0, ze[1] - ze[0])[0]
                else:
                    Q2cur_k[ie, idof]  = 0
                
            if idof in [1, 4] and jdof in [1, 4]:
                rod_i = 0 if idof == 1 else 1
                rod_j = 0 if jdof == 1 else 1
    
                def eqn_M(z):
                    return rA[ie] * N_rod[ie][rod_i](z) * N_rod[ie][rod_j](z)
    
                def eqn_K(z):
                    return EA[ie] * dN_rod[ie][rod_i](z) * dN_rod[ie][rod_j](z) 
                        
    
                M_k_rod[ie, idof, jdof] = scp.quad(eqn_M, ze[0], ze[1])[0]
                K_k_rod[ie, idof, jdof] = scp.quad(eqn_K, ze[0], ze[1])[0]
            
                
            elif idof in beam_dofs and jdof in beam_dofs:
                beam_i = beam_dofs[idof]
                beam_j = beam_dofs[jdof]
    
                def eqn_Mb(z):
                    return rA[ie] * N_k_b[ie][beam_i](z) * N_k_b[ie][beam_j](z) \
                          # - rI[ie] * N_k_b[ie][beam_j](z) * ddN_k_b[ie][beam_i](z)
                
                # def eqn_Mb_Rayleigh(z):
                #     return -rI[ie] * N_k_b[ie][beam_j](z) * ddN_k_b[ie][beam_i](z)
    
                def eqn_Kb(z):
                    return EI[ie] * ddN_k_b[ie][beam_i](z) * ddN_k_b[ie][beam_j](z) \
                          + T_self[ie] * dN_k_b[ie][beam_j](z) * dN_k_b[ie][beam_i](z)
    
                # Destabilizing force due to gravity component in direction of bending
                def eqn_F(z):
                    return rA[ie] * g * dN_k_b[ie][beam_j](z) * N_k_b[ie][beam_i](z)

                def eqn_R(z):
                    return rho_water * g * Rs[ie]**2 * np.pi * dN_k_b[ie][beam_j](z) * N_k_b[ie][beam_i](z)
    
                def eqn_D(z):
                    return 0.5 * rho_water * Cd * np.pi * Rs[ie]**2 * N_k_b[ie][beam_j](z) * N_k_b[ie][beam_i](z)
    
                def eqn_I(z):
                    return rho_water * (1 + Ca) * np.pi * Rs[ie]**2 * N_k_b[ie][beam_j](z) * N_k_b[ie][beam_i](z)
                
                M_k_beam[ie, idof, jdof] = scp.quad(eqn_Mb, 0, ze[1]-ze[0])[0]
                K_k_beam[ie, idof, jdof] = scp.quad(eqn_Kb, 0, ze[1]-ze[0])[0] # + scp.quad(eqn_T, 0, ze[1]-ze[0])[0]
                F_k_beam[ie, idof, jdof] = scp.quad(eqn_F, 0, ze[1]-ze[0])[0]
            
    
                if zn[ie + 1] <= 0:
                    R_k_beam[ie, idof, jdof] = scp.quad(eqn_R, 0, ze[1]-ze[0])[0]
                    D_k_beam[ie, idof, jdof] = scp.quad(eqn_D, 0, ze[1]-ze[0])[0]
                    I_k_beam[ie, idof, jdof] = scp.quad(eqn_I, 0, ze[1]-ze[0])[0]
                    
                else:
                    R_k_beam[ie, idof, jdof] = 0
                    D_k_beam[ie, idof, jdof] = 0
                    I_k_beam[ie, idof, jdof] = 0 


# Add the RNA at the top of the mass matrix
M_k_beam[ne-1][3,3] += m_RNA
M_k_beam[-1][5,5] += I55_RNA
M_k_rod[ne-1][4,4] += m_RNA
D_k_beam[0][1,1] += 130000

# Add restoring coefficient in heave
C33 = rho_water*np.pi*R_buoy[-1]**2*g

# Include the heave hydrostatic restoring coefficient in the K matrix
K_k_rod[0][1,1] += C33

# Here we including the contribution of the RNA to the destabilizing force
# component due to gravity. It is positioned such that the force
F_k_beam[-1][3,5] -= m_RNA*g
F_k_beam[-1][5,3] += m_RNA*g
# F_k_beam[-1][3,3] += m_RNA

# Include the RNA in the restoring term 

#%% Step 4: Global Assembly
ndofs = nn*3
K = np.zeros((ndofs, ndofs))  # Global stiffness matrix
M = np.zeros((ndofs, ndofs))  # Global mass matrix
Q = np.zeros((ndofs, ndofs))  # Global force vector due to gravity
H = np.zeros((ndofs, ndofs))  # Global hydrostatic restoring force
D = np.zeros((ndofs, ndofs))  # Global drag force
I = np.zeros((ndofs, ndofs))  # Added mass force
L = np.zeros((ndofs)) # Current force

for ie in np.arange(0, ne):
    dofs = elem_dofs[ie]
    NodeLeft = dof_node[dofs[0]]
    NodeRight = dof_node[dofs[-1]]

    Dofs_Left = 3 * NodeLeft + np.arange(0, 3)
    Dofs_Right = 3 * NodeRight + np.arange(0, 3)

    nodes = np.append(Dofs_Left, Dofs_Right)

    for i in np.arange(0, 6):
        for j in np.arange(0, 6):
            M[nodes[i], nodes[j]] += M_k_rod[ie, i, j] + M_k_beam[ie, i, j]
            K[nodes[i], nodes[j]] += K_k_rod[ie, i, j] + K_k_beam[ie, i, j] 
            Q[nodes[i], nodes[j]] += F_k_beam[ie, i, j]
            H[nodes[i], nodes[j]] += R_k_beam[ie, i, j]
            D[nodes[i], nodes[j]] += D_k_beam[ie, i, j]
            I[nodes[i], nodes[j]] += I_k_beam[ie, i, j]
            if (i == 0) and (j == 0 or j == 3):
                L[nodes[i]] += Q2cur_k[ie, j]
            else:
                L[nodes[i]] += 0
     
plt.figure()
plt.spy(M)
plt.title('Mass Matrix')

print('M',np.allclose(M, M.T))

plt.figure()
plt.spy(K)
plt.title('Stiffness Matrix')

print('K',np.allclose(K, K.T))

plt.figure()
plt.spy(H)
plt.title('Hydrostatic Restoring Matrix')

print('H',np.allclose(H, H.T))

plt.figure()
plt.spy(D)
plt.title('Hydrodynamic Damping Matrix')

print('D',np.allclose(D, D.T))

plt.figure()
plt.pcolor(np.dot(np.identity(ndofs), L).reshape(-1,1))
plt.title('Current Force Matrix')

# Create a Rayleigh damping matrix
C = 0.01*M + 0.01*K

print('I',np.allclose(I, I.T))

# Update the stiffness matrix with destabilizing and restoring forces
K += -Q + H

print('Q',np.allclose(Q, Q.T))

# Include the added mass matrix I in M
M += I

print('L shape = ',np.shape(L))
print('L = ',L)
print('Sum of current force = ',np.sum(L))

#%% TEST THE DESTABILIZING FORCE
qtest = np.zeros(2*ndofs)
udofs = np.arange(0, ndofs)
vdofs = np.arange(ndofs, 2*ndofs)

du0 = np.deg2rad(5)
qtest[udofs[0::3]] = np.ones_like(udofs[0::3])*du0

F = Q @ qtest[udofs]

print(np.sum(F))

#%% HYDROSTATIC PROPERTIES
print('Hydrostatic Properties:')
Iy = np.pi*R_buoy[-1]**4/4
# V = L*(1-rho_water/rho)*np.pi*RO**2
BM = Iy/V0_buoy

print('BM = ',BM)

KB = cob_buoy_check - draft

print('KB = ',KB)

KG = cog_buoy - draft

print('KG = ',KG)

GM = KB + BM - KG

print('GM = ',GM)

# Pitch natural frequency
C55 = rho_water*g*V0_buoy*GM
I55 = 4229230000 + I55_RNA + m_RNA*(90-cog_buoy)**2
wn5 = np.sqrt(C55/I55) # [rad/s]
f5 = wn5/(2*np.pi)

# Heave natural frequency
wn3 = np.sqrt(C33/m_buoy) # [rad/s]
f3 = wn3/(2*np.pi)

print('C_33 = ',C33)
print('Pitch Natural Frequency = ',f5,'Hz')
print('Pitch Natural Period = ',1/f5,'s')
print('Heave Natural Frequency = ',f3, 'Hz')
print('Heave Natural Period = ',1/f3, 'Hz')
#%% DEFINE INITIAL CONDITIONS

# Initial position = 0
u0 = 0  # [m]

# Initial acceleration = 0
u0_dt2 = 0  # [m/s^2]

# Initial angle = 0
du0 = np.deg2rad(0)  # [m]

# Initial angular acceleration = 0
du0_dt2 = 0  # [m/s^2]

#%% DEFINE A STATE VECTOR
udofs = np.arange(0, ndofs)
vdofs = np.arange(ndofs, 2*ndofs)
q0 = np.zeros(2*ndofs)

q0[udofs[2::3]] = du0
q0[udofs[0::3]] = du0 * zn - (du0 * zn)[0]

#%% SOLVE THE ODE
tf = timeInfo['TDur'] - 1
tspan = np.arange(0, tf, 0.05)

#%% Get the wave velocities at the nodal heights
# zn_submerged = zn <= 0
zn_sub = zn[zn <= 0]
nodal_uwave = np.zeros([len(waves['t']),len(zn_sub)])

i = 0
for t in waves['t']:
    
    nodal_uwave[i,:] = np.interp(zn_sub, waves['z'], waves['u'][i])
        
    i += 1

plt.figure()
plt.plot(zn_sub, nodal_uwave[0::10,:].T)

def uwave(t):
    return np.array([np.interp(t, waves['t'], nodal_uwave[:, i]) for i in range(len(zn_sub))])

plt.figure()
for t in np.arange(0, 50):
    plt.plot(zn_sub, uwave(t))
    
plt.grid()

idx = vdofs[0::3][0:len(zn_sub)]

qwave = np.zeros(2*ndofs)

plt.figure()
plt.plot(waves['t'], waves['eta'])

#%% Calculate the pressure at the base of the buoy
nodal_pressure = np.zeros(len(waves['t']))
i = 0
for t in waves['t']:
    nodal_pressure[i] = np.interp(draft, waves['z'], waves['phi'][i])
    i += 1
    
plt.figure()
plt.plot(waves['t'],nodal_pressure)

def pbase(t):
    return np.interp(t, waves['t'], nodal_pressure)

#%% Get the wind velocity\
def V_hub(t):
    return np.interp(t, wind['t'], wind['V_hub'])

#%% SOLVE THE ODE
Fz = np.zeros_like(q0[udofs])
Fw = np.zeros_like(q0[vdofs])
Fm = np.zeros_like(q0[vdofs])

R = np.identity(ndofs)

#%% FIND EQUILIBRIUM WITH CURRENT
vrel0 = q0[vdofs[-3]] - V_hub(0)
Fw[-3] = np.interp(vrel0, thrust_table['x'], thrust_table['y'])*1e3
cur_equ = np.linalg.solve(K, +np.dot(np.identity(ndofs), L).reshape(-1,1) \
         + Fw.reshape(-1,1))
# q0[0:ndofs] = cur_equ.flatten()
    

#%% DEFINE THE ODE SOLVER
def odefun(t, q):             
    
    # Get the wave velocity at the underwater nodes
    qwave[idx] = uwave(t)
    
    # Calculate the relative velocity
    urel = q[vdofs] - qwave[vdofs]
    
    vrel = q[vdofs[-3]] - V_hub(t)
    
    Fz[1] = np.pi*RO_buoy[0]**2*pbase(t)
    Fw[-3] = np.interp(vrel, thrust_table['x'], thrust_table['y'])*1e3
    Fmxl, Fmzl, Fmxr, Fmzr = Fmoor(q[udofs][6])
    
    # Add the mooring line forces
    Fm[Fair_idx*3] = Fmxr + Fmxl
    Fm[Fair_idx*3+1] = Fmzl + Fmzr
    Fm[Fair_idx*3+2] = 5.2*Fmzl - 5.2*Fmzr
        
    # rhs = Fwind(t) - (np.dot(Kii, q[udofs].reshape(-1, 1))).flatten()
    ud = np.linalg.solve(M, -(
            + Fz.reshape(-1,1) \
            - Fw.reshape(-1,1) \
            + np.dot(D, (urel.reshape(-1,1))*np.abs((urel.reshape(-1,1)))) \
            - np.dot(R, L).reshape(-1,1) \
            + np.dot(K, q[udofs].reshape(-1,1)) \
            - Fm.reshape(-1,1) \
            + np.dot(C, q[vdofs].reshape(-1,1)) \
            ).flatten())
    return np.append(q[vdofs], ud)

sol = scp.solve_ivp(fun=odefun, t_span=[tspan[0], tspan[-1]], y0=q0, 
                    t_eval=None, method='Radau',max_step=0.1)

#%% PLOT THE RESULTS

# Create a figure with four vertically stacked subplots
fig, axs = plt.subplots(4, 1, figsize=(18, 12))

# Plot u deflection
axs[0].plot(sol.t, sol.y[udofs[0]], label='Base of Buoy')
axs[0].plot(sol.t, sol.y[udofs[-3]], label='RNA')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel("Deflection [m]")
axs[0].set_title("Beam deflection")
axs[0].grid()
axs[0].legend(loc='upper right')

# Plot pitch
axs[1].plot(sol.t, np.rad2deg(sol.y[udofs[2]]), label='Base of Buoy')
axs[1].plot(sol.t, np.rad2deg(sol.y[udofs[-1]]), label='RNA')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel("Angular deflection [deg]")
axs[1].set_title("Beam angular deflection")
axs[1].grid()
axs[1].legend(loc='upper right')

# Plot heave
axs[2].plot(sol.t, sol.y[udofs[1]], label='Base of Buoy')
axs[2].plot(sol.t, sol.y[udofs[-2]], label='RNA')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel("Excursion [m]")
axs[2].set_title("Rod Heave")
axs[2].grid()
axs[2].legend(loc='upper right')

# Plot wind speed and wave elevation
axs[3].plot(waves['t'], waves['eta'], linewidth=0.7, label='Wave Elevation', color='blue')
axs[3].plot(wind['t'], wind['V_hub'], linewidth=0.7, label='Wind Speed', color='orange')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel("Magnitude")
axs[3].set_title("Environmental Conditions")
axs[3].grid()
axs[3].legend(loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined plot
plt.show()

#%% Run an FFT on Pitch
from scipy.fft import fft, fftfreq
n = len(sol.t)
yf = fft(sol.y[udofs[-1]])
xf = fftfreq(n, sol.t[1] - sol.t[0])

# Plotting the frequency response
plt.figure(figsize=(10, 5))
plt.semilogy(xf/(2*np.pi), np.abs(yf), label='Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Frequency Response of RNA Angular Deflection')
plt.grid()
plt.xlim([0,50])
plt.legend()
plt.show()

#%% Run an FFT on Heave
n = len(sol.t)
yf = fft(sol.y[udofs[-2]])
xf = fftfreq(n, sol.t[1] - sol.t[0])

# Plotting the frequency response
plt.figure(figsize=(10, 5))
plt.semilogy((xf/(2*np.pi)), np.abs(yf), label='Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Frequency Response: Heave')
plt.grid()
plt.xlim([0,100])
plt.legend()
plt.show()

#%% FFT: surge
n = len(sol.t)
yf = fft(sol.y[udofs[-3]])
xf = fftfreq(n, sol.t[1] - sol.t[0])

# Plotting the frequency response
plt.figure(figsize=(10, 5))
plt.semilogy((xf/(2*np.pi)), np.abs(yf), label='Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Frequency Response: Surge')
plt.grid()
plt.xlim([0,50])
plt.legend()
plt.show()

#%% PLOT THE BEAM FOR THE LAST TIME STEP
plt.figure(figsize=(5,10))
# plt.figure()

u = sol.y[udofs[0::3]]
w = sol.y[udofs[1::3]]
phi = sol.y[udofs[2::3]]

tstep = 0
# xpos = u[:,int(tstep)]
# zpos = w[:,int(tstep)] + zn

xpos = u[:,-1]
zpos = w[:,-1] + zn

plt.plot(np.zeros_like(xpos), zn,'o-',label='Equilibrium')
plt.plot(xpos, zpos,'o--',label=f'Time t = {np.round(sol.t[20])} s')
plt.grid()
plt.xlabel('Surge excursion [m]')
plt.ylabel('Nodal position / heave excursion [m]')
plt.legend()

# Plot the pitch-corrected displacement
plt.figure(figsize=(5,10))
xpos = u[:,-1]
zpos = w[:,-1] + zn

# plt.plot(np.zeros_like(xpos), zn,'o-',label='Equilibrium')
plt.plot(xpos-phi[0,-1]*zn, zpos,'o--',label=f'Time t = {np.round(sol.t[20])} s')
plt.grid()
plt.xlabel('Surge excursion [m]')
plt.ylabel('Nodal position / heave excursion [m]')
plt.legend()

end_time = time.time()

#%% PLOT THE ROD DEFLECTION
plt.figure(figsize=(5,10))
# plt.figure()

u = sol.y[udofs[0::3]]
w = sol.y[udofs[1::3]]
phi = sol.y[udofs[2::3]]

tstep = len(sol.t)-1
xpos = u[:,int(tstep)]
zpos = w[:,int(tstep)] + zn

plt.plot(np.zeros_like(xpos), zn,'o-',label='Equilibrium')
# plt.plot(xpos-phi[:,int(tstep)]*zn, zpos,'o--',label=f'Time t = {np.round(sol.t[-1])} s')
plt.plot(xpos, zpos,'o--',label=f'Time t = {np.round(sol.t[-1])} s')

# plt.xlim([-2, 0.5])
plt.grid()
plt.xlabel('Surge excursion [m]')
plt.ylabel('Nodal position / heave excursion [m]')
plt.legend()

# %% MODAL ANALYSIS
mat = np.dot(np.linalg.inv(M), K)
w2, vr = np.linalg.eig(mat)
w = np.sqrt(w2.real)
f = w/2/np.pi
len(f)

idx = f.argsort()#[::-1]  
f = f[idx]
vr_sorted = vr[:,idx]
nDof = nn*6
ModalShape = np.zeros((nDof, len(f)))

plt.figure(figsize=(18,4))
for i in np.arange(0,5):
    plt.plot(zn + vr_sorted[1::3,i],vr_sorted[0::3,i],'-o',label=f'Mode {i+1}: T = {np.round(1/f[i],2)} [s]')
    # plt.plot(np.zeros_like(vr_sorted[0::3,i]), vr_sorted[1::3,i])
plt.grid()
plt.xlabel('Nodal Position Along Tower [m]')
plt.ylabel('Excursion [~]')
plt.title('Beam Modal Shapes')
plt.legend()

# Compute the damping ratio for 25 modes
nMode = nn*3
PHI = vr_sorted[:,0:nMode]

Mm = np.zeros(nMode)
Km = np.zeros(nMode)
Cm = np.zeros(nMode)
ModalDampRatio = 0.01
# Compute your "nMode" entries of the modal mass, stiffness and damping
for iMode in np.arange(0,nMode):
    print('Computing Mode: ',iMode+1) # Starts at 0 off course
    Mm[iMode] = PHI[:,iMode].T @ M @ PHI[:,iMode]
    Km[iMode] = PHI[:,iMode].T @ K @ PHI[:,iMode]
    Cm[iMode] = 0.01*Mm[iMode] + 0.01*Km[iMode]
    print('Mm = ',Mm[iMode],', Km = ', Km[iMode],', Cm = ', Cm[iMode])
    
#%% SOLVE THE MODAL ODE
qwave = np.zeros_like(q0)
idx = vdofs[0::3][0:len(zn_sub)]

# Solve the resulting ODE:
def qdot(t,q):
    
    # Get the wave velocity at the underwater nodes
    qwave[idx] = uwave(t)

    U_F = np.sum(PHI*q[0:nMode],1)
    V_F = np.sum(PHI*q[nMode:2*nMode],1)
    
    # Calculate the relative velocity
    urel = V_F - qwave[vdofs]
    
    vrel = V_F[-3] - V_hub(t)
    
    Fz[1] = np.pi*RO_buoy[0]**2*pbase(t)
    Fw[-3] = np.interp(vrel, thrust_table['x'], thrust_table['y'])*1e3
    Fmxl, Fmzl, Fmxr, Fmzr = Fmoor(U_F[6])
    Fm[Fair_idx*3] = Fmxr + Fmxl
    Fm[Fair_idx*3+1] = Fmzl + Fmzr
    Fm[Fair_idx*3+2] = 5.2*Fmzl - 5.2*Fmzr
    
    F = - Fz \
        + Fw \
        - np.dot(D, (urel * np.abs(urel))) \
        + L \
        + Fm \
        
    Am = ( PHI.T @ F - (Km * q[udofs] + Cm * q[vdofs]) ) / Mm
    return np.append(q[vdofs],Am)

q0 = np.zeros(2*nMode)

q = scp.solve_ivp(fun=qdot, t_span=[tspan[0], tspan[-1]], y0=q0, 
                    t_eval=None, method='Radau',max_step=0.1)

plt.figure()
U_F = np.zeros((len(q.t),len(PHI[:,0])))

# For every mode shape
for iMode in np.arange(0,nMode):
    
    # For every time step
    for it in np.arange(0,len(q.t)):
        
        # The displacement at time step it is equal to:
        U_F[it,:] += PHI[:,iMode] * q.y[iMode][it]
  
#%% PLOT THE MODAL AND FEM RESULTS FOR TIME HISTORY

# Create a figure with three vertically stacked subplots
fig, axs = plt.subplots(4, 1, figsize=(18, 12))

# Plot the excursion on the first subplot
axs[0].plot(q.t, U_F[:, -3], label='Modal: RNA', color='blue')
axs[0].plot(sol.t, sol.y[udofs[-3]], label='FEM: RNA', color='green')
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Excursion [m]")
axs[0].set_title("Surge")
axs[0].legend(loc='upper left')
axs[0].grid()

# Plot the excursion on the first subplot
axs[1].plot(q.t, U_F[:, -2], label='Modal: RNA', color='blue')
axs[1].plot(sol.t, sol.y[udofs[-2]], label='FEM: RNA', color='green')
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Excursion [m]")
axs[1].set_title("Heave")
axs[1].legend(loc='upper left')
axs[1].grid()

# Plot the excursion on the first subplot
axs[2].plot(q.t, U_F[:, -1], label='Modal: RNA', color='blue')
axs[2].plot(sol.t, sol.y[udofs[-1]], label='FEM: RNA', color='green')
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("Excursion [m]")
axs[2].set_title("Pitch")
axs[2].legend(loc='upper left')
axs[2].grid()

# Plot wave elevation and wind speed on the second subplot with two y-axes
# Plot wind speed and wave elevation
axs[3].plot(waves['t'], waves['eta'], linewidth=0.7, label='Wave Elevation', color='blue')
axs[3].plot(wind['t'], wind['V_hub'], linewidth=0.7, label='Wind Speed', color='orange')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel("Magnitude")
axs[3].set_title("Environmental Conditions")
axs[3].grid()
axs[3].legend(loc='upper right')

plt.tight_layout()
plt.show()


#%% PITCH FREE DECAY
# DEFINE INITIAL CONDITIONS

# Initial position = 0
u0 = 0  # [m]

# Initial acceleration = 0
u0_dt2 = 0  # [m/s^2]

# Initial angle = 0
du0 = np.deg2rad(5)  # [m]

# Initial angular acceleration = 0
du0_dt2 = 0  # [m/s^2]

# DEFINE A STATE VECTOR
udofs = np.arange(0, ndofs)
vdofs = np.arange(ndofs, 2*ndofs)
q0 = np.zeros(2*ndofs)

q0[udofs[2::3]] = du0
q0[udofs[0::3]] = du0 * zn - (du0 * zn)[0]


print(np.sum(Q @ q0[udofs]))
#%%

def odefun_pitch_decay(t, q):             
    
    # Calculate the relative velocity
    urel = q[vdofs] #- qwave[vdofs]
    
    Fmxl, Fmzl, Fmxr, Fmzr = Fmoor(q[udofs][6])
    
    # Add the mooring line forces
    Fm[Fair_idx*3] = Fmxr + Fmxl
    Fm[Fair_idx*3+1] = Fmzl + Fmzr
    Fm[Fair_idx*3+2] = 5*Fmzl - 5*Fmzr
        
    ud = np.linalg.solve(M, -(
            # + Fz.reshape(-1,1) \
            # - Fw.reshape(-1,1) \
            + np.dot(D, (urel.reshape(-1,1))*np.abs((urel.reshape(-1,1)))) \
            # - np.dot(R, L).reshape(-1,1) \
            + np.dot(K, q[udofs].reshape(-1,1)) \
            - Fm.reshape(-1,1) \
            + np.dot(C, q[vdofs].reshape(-1,1)) \
            ).flatten())
    return np.append(q[vdofs], ud)

sol_pitch_decay = scp.solve_ivp(fun=odefun_pitch_decay, t_span=[0, 200], y0=q0, 
                    t_eval=None, method='Radau',max_step=0.1)


#%% HEAVE FREE DECAY
# DEFINE INITIAL CONDITIONS

# Initial position = 0
w0 = 1  # [m]

# Initial acceleration = 0
w0_dt2 = 0  # [m/s^2]

# Initial angle = 0
du0 = np.deg2rad(0)  # [m]

# Initial angular acceleration = 0
du0_dt2 = 0  # [m/s^2]

# DEFINE A STATE VECTOR
udofs = np.arange(0, ndofs)
vdofs = np.arange(ndofs, 2*ndofs)
q0 = np.zeros(2*ndofs)

q0[udofs[1::3]] = w0
# q0[udofs[0::3]] = du0 * zn - (du0 * zn)[0]

print(q0)

def odefun_heave_decay(t, q):             
    
    # Calculate the relative velocity
    urel = q[vdofs] #- qwave[vdofs]
    
    Fmxl, Fmzl, Fmxr, Fmzr = Fmoor(q[udofs][6])
    
    # Add the mooring line forces
    Fm[Fair_idx*3] = Fmxr + Fmxl
    Fm[Fair_idx*3+1] = Fmzl + Fmzr
    Fm[Fair_idx*3+2] = 5.2*Fmzl - 5.2*Fmzr
        
    ud = np.linalg.solve(M, -(
            # + Fz.reshape(-1,1) \
            # - Fw.reshape(-1,1) \
            + np.dot(D, (urel.reshape(-1,1))*np.abs((urel.reshape(-1,1)))) \
            # - np.dot(R, L).reshape(-1,1) \
            + np.dot(K, q[udofs].reshape(-1,1)) \
            - Fm.reshape(-1,1) \
            + np.dot(C, q[vdofs].reshape(-1,1)) \
            ).flatten())
    return np.append(q[vdofs], ud)

sol_heave_decay = scp.solve_ivp(fun=odefun_pitch_decay, t_span=[0, 200], y0=q0, 
                    t_eval=None, method='Radau',max_step=0.1)

#%% PLOT THE FREE DECAY RESULTS

# Create a figure with 3x2 subplots
fig, axs = plt.subplots(3, 2, figsize=(20, 8))

# Define the data and titles for each subplot
data = [
    (sol_heave_decay, "Free Decay Responses: Initial Heave Excursion"),
    (sol_pitch_decay, "Free Decay Responses: Initial Pitch Excursion")
]

titles = [
    ["Beam excursion: surge", "Beam excursion: pitch", "Rod excursion: heave"],
    ["Beam excursion: surge", "Beam excursion: pitch", "Rod excursion: heave"]
]

# Loop through the data and create plots
for col, (sol_free, suptitle) in enumerate(data):
    # Plot u deflection
    axs[0, col].plot(sol_free.t, sol_free.y[udofs[0]], label='Base of Buoy')
    axs[0, col].plot(sol_free.t, sol_free.y[udofs[-3]], label='RNA')
    axs[0, col].set_xlabel('Time [s]')
    axs[0, col].set_ylabel("Deflection [m]")
    axs[0, col].set_title(titles[col][0])
    axs[0, col].grid()
    axs[0, col].legend(loc='upper right')

    # Plot pitch
    axs[1, col].plot(sol_free.t, np.rad2deg(sol_free.y[udofs[2]]), label='Base of Buoy')
    axs[1, col].plot(sol_free.t, np.rad2deg(sol_free.y[udofs[-1]]), label='RNA')
    axs[1, col].set_xlabel('Time [s]')
    axs[1, col].set_ylabel("Angular deflection [deg]")
    axs[1, col].set_title(titles[col][1])
    axs[1, col].grid()
    axs[1, col].legend(loc='upper right')

    # Plot heave
    axs[2, col].plot(sol_free.t, sol_free.y[udofs[1]], label='Base of Buoy')
    axs[2, col].plot(sol_free.t, sol_free.y[udofs[-2]], label='RNA')
    axs[2, col].set_xlabel('Time [s]')
    axs[2, col].set_ylabel("Deflection [m]")
    axs[2, col].set_title(titles[col][2])
    axs[2, col].grid()
    axs[2, col].legend(loc='upper right')

fig.suptitle('Free Decay Responses: 1 m Heave (left), 5 deg Pitch (right)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print('Elapsed = ',end_time - start_time)

