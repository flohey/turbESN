###############################################################################################
# This script runs ensemble grid/random searches
###############################################################################################
# DEVICE: CPU
###############################################################################################
# Florian Heyder - 15.06.2021
###############################################################################################

#Parallelization
import multiprocessing as mp
from concurrent import futures

#Misc. 
import sys
import os
import time
from copy import deepcopy

#Backends
import numpy as np
import torch

#Data structure
import h5py

#turbESN
from turbESN.util import (
    PreparePredictorData,   
    InitStudyOrder,
    CreateHDF5Groups, 
    InitRandomSearchStudyOrder
    )
from turbESN.core import *
from turbESN.study import launch_thread_RunturbESN, launch_process_RunturbESN, Callback


from sklearn.preprocessing import minmax_scale


MAX_PROC = 4                                                          #no. subprocesses (here: each running a seed)
MAX_THREADS = 1                                                       #max. no. threads per subprocess (if not using a full machine: set to 1)

if __name__ == '__main__':    
    #----------------------------------------------------
    #PATH PARAMETERS
    # - specify where the study should be saved to
    # - specify where the data is located
    #----------------------------------------------------
    path_data = '/home/flhe/Documents/hdf5/Lorenz63/data/'                           #path to hdf5 data used for ESN (see ImportData function)
    filename_data = 'Lorenz63.hdf5'

    path_esn = '/home/flhe/'                             #path to where hdf5 ESN results should be saved to
    filename_esn = 'Test.hdf5'
    filepath_esn = path_esn + filename_esn

    #----------------------------------------------------
    #STUDY PARAMETERS
    # - set list of random seeds
    # - choose grid or random search mode
    # - specify hyperparameters to study and their range
    #----------------------------------------------------
    randomSeed = range(5)                                               #random number generator seeds that will be used to initialize the reservoir
    nseed = len(randomSeed)         

    doRandomSearch = False                                                #if not RandomSearch then do GridSearch
    study_tuple = ('spectralRadius', 'reservoirDensity')                  #set with names of parameters which should be studied (same order as study_parameters)

    #----------------------------------------------------                  
    if doRandomSearch:
        nstudy = 1000
        config = InitRandomSearchStudyOrder(nstudy, study_tuple)

        print('Random Search. Seeds: {0}. Studies per seed: {1}\n'.format(nseed, nstudy))
    
    else:
        spectralRadius_array   = np.linspace(0.1,2,2)
        reservoirDensity_array = np.linspace(0.1,1,2)
        study_parameters = (spectralRadius_array,reservoirDensity_array)           #tuple specifying the study parameter range. 

        nstudyparameters = len(study_parameters)
        assert nstudyparameters == len(study_tuple),'''Error: Length of study_tuple ({0}) 
                                                       does not match no. study parameters ({1})!\n'''.format(len(study_tuple), nstudyparameters)

        nstudy = np.prod([len(param_arr) for param_arr in study_parameters])
        config = InitStudyOrder(nstudy, study_parameters) 
        print('Grid Search. Seeds: {0}. Studies per seed: {1}\n'.format(nseed, nstudy))
    
    #----------------------------------------------------
    #DATA PARAMETERS
    # - import data 
    # - data shape: (timesteps, n_input)
    #----------------------------------------------------
    with h5py.File(path_data + filename_data,'r') as f:
        data = np.array(f.get('data'))
    
    data_scaled = minmax_scale(data, axis = 0)   #scale to [0,1] along time axis
    n_input_data = data.shape[1]
    data_timesteps = data.shape[0]     


    trainingLength = 3050
    testingLength = 1000
    validationLength = 1000
    transientTime = 50
    esn_start = data_timesteps - int(trainingLength + testingLength + validationLength)
    esn_end = data_timesteps
    #----------------------------------------------------
    #ESN PARAMETERS
    # - create ESN
    # - adapt ESN setting
    # - adapt ESN training and testing data
    #----------------------------------------------------
    n_input = n_input_data
    data_scaled = data_scaled[:,:n_input]

    n_reservoir = 512
    reservoirDensity = 0.2
    regressionParameter = 5e-1
    leakingRate = 1.0
    spectralRadius = 0.95
    
    esn = ESN(randomSeed = randomSeed[0],
           esn_start = esn_start,
           esn_end = esn_end,
           trainingLength = trainingLength,
           testingLength = testingLength,
           validationLength = validationLength,
           data_timesteps = data_timesteps,
           n_input = n_input,
           n_output = n_input,
           n_reservoir = n_reservoir,
           leakingRate = leakingRate,
           spectralRadius = spectralRadius,
           reservoirDensity = reservoirDensity,
           regressionParameter = regressionParameter,
           bias_in = 1,
           bias_out = 1,
           outputInputScaling = 0.0,
           inputScaling = 1.0,
           inputDensity = 1.0,
           noiseLevel_in = 1e-6,
           noiseLevel_out = 0.0,
           mode = 'auto',
           weightGeneration = 'uniform',
           extendedStateStyle = 'default',
           transientTime  = transientTime, 
           use_watts_strogatz_reservoir = False,
           ws_p = 0.3,
           verbose = False)


    u_train, y_train, u_test, y_test, u_val, y_val = PreparePredictorData(data_scaled, 
                                                            n_input=esn.n_input, 
                                                            trainingLength=esn.trainingLength, 
                                                            testingLength=esn.testingLength, 
                                                            validationLength=esn.validationLength,
                                                            esn_start=esn.esn_start, 
                                                            esn_end=esn.esn_end)


                                                            
    esn.SetTrainingData(u_train, y_train)
    esn.SetTestingData(y_test, pred_init_input=y_train[-1,:])     
    esn.SetValidationData(y_val=y_val, u_val=u_val)     
    

    #----------------------------------------------------
    #RUN STUDY 
    # - copy ESN parameters
    # - change seed
    # - distribute different seeds among processes
    # - distribute different ESN settings among threads
    #----------------------------------------------------
    esn.toTorch()
    esn.save(filepath_esn)
    CreateHDF5Groups(filepath_esn, randomSeed, nstudy)

    time_start = time.time()
    pool = mp.Pool(processes=MAX_PROC)
    with h5py.File(filepath_esn,'a') as f: 
        for esn_id,seed in enumerate(randomSeed):
        
            esn_copy = deepcopy(esn)
            esn_copy.SetRandomSeed(seed)
            esn_copy.SetID(esn_id)

            pool.apply_async(launch_process_RunturbESN, args = ((esn_copy, filepath_esn, MAX_THREADS, config, nstudy, study_tuple),), callback = Callback)       
    
        pool.close()
        pool.join()

    time_end = time.time()
    print('\n ----------------------------------------')
    print('\n PROGRAM FINISHED!')
    print('\n ----------------------------------------')
    print('\n Total elapsed time {0:.2f}s'.format(time_end-time_start))

#---------------------------------------------------------------------------------------------
#                          END OF PROGRAM
#---------------------------------------------------------------------------------------------
   
