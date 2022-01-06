#turbESN
from .util import (
    SaveStudy, 
    RunturbESN,
    ComputeWassersteinDistance
    )

from .core import (ESN, _DTYPE, _DEVICE, _ESN_MODES, _WEIGTH_GENERATION, _LOGGING_FORMAT)

#Parallelization
import multiprocessing as mp
from concurrent import futures

#Misc. 
import sys
import os
import time
from copy import deepcopy
from typing import Union, Tuple, List
import logging

#Backends
import numpy as np
import torch

#Data structure
import h5py

_ID_PRINT = 4
#--------------------------------------------------------------------------------------------------------
def LaunchSingleThread(thread_args):
    ''' Runs an ESN with given esn.
    INPUT:
        esn    - ESN class object containing the reservoir parameters and training & validation/testing data sets.
        config_istudy - study parameter configuration for this ESN run
        study_tuple   - set containing the names of the parameter that are studied
        istudy        - study ID

    RETURN:
        istudy     - Study ID
        mse_train  - mean square error of (teacher forced!) reservoir output (to training target data set y_train) in the training phase. Mean w.r.t. timestep- & mode-axis. 
        mse_test   - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep- & mode-axis. 
        y_pred     - reseroir outputs, produced by the given reservoir specified in esn.
        study_dict - dict specifying the new hyperparameter setting of the ESN 
         
     '''

    esn, config_istudy, study_tuple, istudy, nstudy = thread_args
    torch.manual_seed(esn.randomSeed)

    study_dict = esn.SetStudyParameters(config_istudy, study_tuple)
    mse_train, mse_test, y_pred = RunturbESN(esn)
    #wd_test = ComputeWassersteinDistance(esn.y_test, y_pred)

    if esn.id % _ID_PRINT == 0: 
        logging.warn("ID {0}: study {1}/{2} done.".format(esn.id, istudy+1, nstudy))

    return (istudy, mse_train.numpy(), mse_test.numpy(), y_pred.numpy(), study_dict)
#-------------------------------------------------------------------------------------------------------- 
def LaunchThreads(launch_thread_args) -> Tuple[int, list, int, str]:
    ''' Launches ThreadPool with maximum of MAX_THREADS threads Each thread runs an ESN setting (same RNG seed). 
        The ThreadPool is finished when all studies have been run. 
    
    INPUT:
        esn           - ESN object 
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to 
        MAX_THREADS   - max. no. threads per process
        config        - study configuration
        nstudy        - no. studies
        study_tuple   - tuple specifying the study parameters
    RETURN:
        esn_id        - RNG seed of the ESN
        study_results - results of all studies (same RNG seed). Final shape: [RESULTS RUN1, ..., RESULTS RUN nstudy], 
                        where all RESULTS has shape (istudy, mse_train, mse_test, y_pred, study_dict)
        nstudy        - no. studies
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to 
    '''

    esn, filepath_esn, MAX_THREADS, config, nstudy, study_tuple = launch_thread_args

    study_results = []
    logging.warn('Subprocess {0} starting.'.format(esn.id))
    
    #----------------------------------
    #Set Seed
    #----------------------------------
    torch.manual_seed(esn.randomSeed)             

    #----------------------------------
    #Launch ThreadPool
    #----------------------------------
    
    executor = futures.ThreadPoolExecutor(MAX_THREADS)
    with executor as ex:
        study_counter = 0
        while study_counter < nstudy:   

            if (nstudy-study_counter) < MAX_THREADS:
                num_threads  = nstudy-study_counter
            else:
                num_threads = MAX_THREADS

            thread_results = ex.map(LaunchSingleThread, ([deepcopy(esn), config[istudy], study_tuple, istudy, nstudy] 
                                                 for istudy 
                                                 in np.arange(study_counter,study_counter+num_threads))
                                    )
            study_results.append(list(thread_results))
            study_counter += num_threads

   
    logging.warn('ID {0} all done.'.format(esn.id))
    
    #----------------------------------
    #Reshape result list
    #----------------------------------
    
    study_results = [result for chunks in study_results for result in chunks]   

    return (esn.id, study_results, nstudy, filepath_esn)
#--------------------------------------------------------------------------------------------------------
def Callback(callback_args):
    ''' Callback to main process. Saves results of each seed into open hdf5 file f.
    
    INPUT:
        esn_id        - RNG seed of the ESN runs
        study_results - results of all studies (same RNG seed). Final shape: [RESULTS RUN1, ..., RESULTS RUN nstudy], 
                        where all RESULTS has shape (istudy, mse_train, mse_test, y_pred, study_dict)
        nstudy        - no. studies
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to
    '''
    
    esn_id, study_results, nstudy, filepath_esn = callback_args

    if esn_id % _ID_PRINT == 0:
        logging.debug('ID {0}: returning to main process for saving.'.format(esn_id))

    for ii in range(nstudy):
        istudy, mse_train, mse_test, y_pred, study_dict= study_results[ii] 
        SaveStudy(filepath = filepath_esn, esn_id = esn_id, studyID = istudy,  
                  study_dict = study_dict, y_pred = y_pred, 
                  mse_train = mse_train, mse_test = mse_test) 
        
    if esn_id % _ID_PRINT == 0:        
        logging.warn('Saved study for ID {0}.'.format(esn_id))
#--------------------------------------------------------------------------------------------------------
