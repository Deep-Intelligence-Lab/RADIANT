# Script for generating the synthetic Temporal signals.
# For even distribution of the Heart Rates in the temporal signals, 
# equal number of temporal signals are generated for a HR range of 10 BPM.

import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from scipy.signal import detrend
from tqdm import tqdm
import os

# Generate synthetic temporal signals between the given frequency range [a, b].
def get_sample(a, b):
    """_summary_

    Args:
        a (float): The lower bound of the HR frequency range.
        b (float): The upper bound of the HR frequency range.

    Returns:
        Dictionary: Containing the Temporal map with key "map", Heart rate with key "HR" and Pulse signal with key "pulse".
    """    

    sRate = 20 # Sampling rate in Hertz.
    numPeriods = 26 # Number of periods of the sine waves in the video.
    numSamples = sRate * numPeriods # Total number of frames.

    freq1 = random.uniform(a, b)  # Heart rate frequency.
    hr = freq1*60
    freq2 = random.uniform(0.084,0.34) # Respiratory rate frequency.

    tmap = np.empty((1,40,520)) # For storing the temporal map.

    # Generate temporal map considering 40 facial ROIs.
    for roi in range(40):
    M1 = random.uniform(0, 1) # Amplitude of the heart rate wave.
    M2 = random.uniform(0, 1) # Amplitude of the respiratory rate wave.
    phi = random.uniform(0, 2*3.14) # The phase for the heart rate.
    theta = random.uniform(0, 2*3.14) # The phase for the respiratory rate.
    GN = np.random.normal(0,0.1,size=numSamples) # Gaussian noise.

    def step(t):
        if t<0:
        return 0
        else:
        return 1

    # Create the indices for the temporal signals (an array from 0 upto the total number of samples in the temporal signal).
    x = np.linspace(0, numPeriods, numSamples)

    # Functions to generate the waves.
    f1 = lambda x: M1*np.sin(freq1*2*np.pi*x+phi)
    f3 = lambda x: 0.5*M1*np.sin(2*freq1*2*np.pi*x+phi)
    f2 = lambda x: M2*np.sin(freq2*2*np.pi*x+theta)

    # Generate temporal signal.
    sampled_f1 = [f1(x[i])+f2(x[i])+f3(x[i])+ GN[i] + 0.6*step(x[i]-60)+0.4*step(x[i]-18) + offset for i in range(len(x))]
    # Generate pulse signal.
    bvp = [f1(x[i])+f2(x[i]) for i in range(len(x))]
    # Store the temporal signals into a temporal map.
    tmap[0,roi] = sampled_f1

    # np.save("syn_v1/train_offset/"+str(sample_no)+".npy",tmap)
    # np.save("syn_v1/train_offset/bvp/"+str(sample_no)+".npy", bvp)
    # np.save("syn_v1/train_offset/hr/"+str(sample_no)+".npy", hr)

    return {"map": tmap, "HR": hr, "pulse" : bvp}

if __name__=="__main__":
    
    BASEDIR = os.getcwd()
    indice = BASEDIR.rfind("/")
    BASEDIR = BASEDIR[0:indice]
    
    # Directory for saving the synthetic temporal signals.
    savedir = os.path.join(BASEDIR, "input", "synthetic_temporal_signals")
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    
    
    # The bounds for the frequencies for generating the temporal signals.
    lower_bound = [0.67428571, 0.76357143, 0.85285714, 0.94214286, 1.03142857,
        1.12071429, 1.21      , 1.29928571, 1.38857143, 1.47785714,
        1.56714286, 1.65642857, 1.74571429, 1.835     , 1.92428571,
        2.01357143, 2.10285714, 2.19214286, 2.28142857, 2.37071429,
        2.46      , 2.54928571, 2.63857143, 2.72785714, 2.81714286,
        2.90642857, 2.99571429, 3.085     , 3.17428571, 3.26357143,
        3.35285714, 3.44214286, 3.53142857, 3.62071429, 3.71      ,
        3.79928571, 3.88857143, 3.97785714, 4.06714286, 4.15642857]

    upper_bound = [0.75428571, 0.84357143, 0.93285714, 1.02214286, 1.11142857,
        1.20071429, 1.29      , 1.37928571, 1.46857143, 1.55785714,
        1.64714286, 1.73642857, 1.82571429, 1.915     , 2.00428571,
        2.09357143, 2.18285714, 2.27214286, 2.36142857, 2.45071429,
        2.54      , 2.62928571, 2.71857143, 2.80785714, 2.89714286,
        2.98642857, 3.07571429, 3.165     , 3.25428571, 3.34357143,
        3.43285714, 3.52214286, 3.61142857, 3.70071429, 3.79      ,
        3.87928571, 3.96857143, 4.05785714, 4.14714286, 4.23642857]

    # Total samples generated.
    samples_gen = 0
    
    # Total number of samples.
    N = 5000
    
    # Total number of frequecy ranges.
    total_ranges = len(lower_bound)
    ws = N // total_ranges # Total number of samples in a frequency range.
    
    # Variable to handle the HR frequency ranges. 
    bound = 0
    
    # Generate temporal signals with frequencies in the range given by lower_bound[bound], upper_bound[bound].
    # For each range, N // total_ranges samples are generated for even distribution in the entire HR range.
    for i in tqdm(range(N)):
        if i%ws==0:
            a, b = lower_bound[bound], upper_bound[bound]
            bound += 1
            
        sample = get_sample(a, b)
        samples_gen += 1
        
        
          