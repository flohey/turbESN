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
from turbESN.cross_validation import *
from turbESN.core import *
from turbESN.study import launch_thread_forward_validate_turbESN, launch_process_forward_validate_turbESN, Callback


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
    
    data = minmax_scale(data, axis = 0)   #scale to [0,1] along time axis
    n_input_data = data.shape[1]
    data_timesteps = data.shape[0]     
    #----------------------------------------------------
    #ESN PARAMETERS
    # - create ESN
    # - adapt ESN setting
    #----------------------------------------------------
    n_reservoir = 512

    esn = ESN.L63Reservoir(data_timesteps = data_timesteps, 
                          trainingLength = 3000,
                          testingLength = 1000,
                          validationLength = 1000,
                          mode = 'auto',
                          verbose = False)
    esn.SetRandomSeed(0)
    esn.SetNInputOutput(n_input_data)
    esn.SetNReservoir(n_reservoir)

    #----------------------------------------------------
    #Cross Validation PARAMETERS
    # - specify cross validation parameters
    # - create Cross Validation object
    #----------------------------------------------------
    esn_start = 1
    esn_end = data_timesteps
    
    n_folds = 8
    fold_length = 400

    n_training_folds = 2
    n_testing_folds = 1
    n_validation_folds = 1

    cv = CrossValidation(data, 
                         esn=esn,
                         n_folds=n_folds,
                         fold_length=fold_length, 
                         n_training_folds=n_training_folds,
                         n_testing_folds=n_testing_folds, 
                         n_validation_folds=n_validation_folds)

    #----------------------------------------------------
    #RUN STUDY 
    # - copy ESN parameters
    # - change seed
    # - distribute different seeds among processes
    # - distribute different ESN settings among threads
    #----------------------------------------------------
    esn.toTorch()
    esn.save(filepath_esn)
    cv.save(filepath_esn)
    CreateHDF5Groups(filepath_esn, randomSeed, nstudy)

    time_start = time.time()
    pool = mp.Pool(processes=MAX_PROC)
    with h5py.File(filepath_esn,'a') as f: 
        for esn_id,seed in enumerate(randomSeed):
        
            esn_copy = deepcopy(esn)
            esn_copy.SetRandomSeed(seed)
            esn_copy.SetID(esn_id)
            cv_copy = deepcopy(cv)

            pool.apply_async(launch_process_forward_validate_turbESN, args = ((esn_copy, cv_copy, filepath_esn, MAX_THREADS, config, nstudy, study_tuple),), callback = Callback)       
    
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
   
