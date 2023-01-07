#turbESN
from .util import (
    save_study, 
    run_turbESN,
    forward_validate_auto_ESN,
    compute_wasserstein_distance,
    compute_mse,
    compute_nrmse
    )

from .core import ESN
from .cross_validation import CrossValidation
from ._modes import (_DTYPE, _DEVICE, _ESN_MODES, _WEIGTH_GENERATION, _EXTENDED_STATE_STYLES, _ID_PRINT)

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
import logging.config

#Backends
import numpy as np
import torch

#Data structure
import h5py
import json

# Read hyperparameter.json
import importlib.resources as pkg_resources
with pkg_resources.path(__package__,'hyperparameters.json') as hp_dict_path:
    with open(hp_dict_path,'r') as f:
        HP_dict = json.load(f)   
with pkg_resources.path(__package__,'logging_config.json') as logging_config_path:
    with open(logging_config_path,'r') as f:
        LOGGING_CONFIG = json.load(f)

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('turbESNlogger')
#--------------------------------------------------------------------------------------------------------
def thread_run_turbESN(thread_args) -> Tuple[int,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,dict]:
    ''' Runs an ESN with given esn.
    INPUT:
        esn                - ESN class object containing the reservoir parameters and training & validation/testing data sets.
        config_istudy      - study parameter configuration for this ESN run
        study_tuple        - set containing the names of the parameter that are studied
        istudy             - study identifier (can either be current ESN setting identifier (parallelize seeds) or ESN random seed identifier (parallelize settings))
        nstudy             - no. studies (either no. settings (nsetting) or no. RNG seeds (nseed))
        recompute_Win      - whether input matrix must be initialized
        recompute_Wres     - whether reservoir matrix must be initialized
        recompute_Wfb      - whether feedback matrix must be initialized

    RETURN:
        istudy     - study ID
        mse_train  - mean square error of (teacher forced) reservoir output (to training target data set y_train). Mean w.r.t. timestep-axis. 
        mse_test   - mean square error of reservoir output (to testing data set y_test). Mean w.r.t. timestep-axis. 
        mse_val    - mean square error of reservoir output (to validation data set y_val). Mean w.r.t. timestep-axis. 
        y_pred_test- testing reseroir outputs, produced by the given reservoir specified in esn.
        y_pred_val - validation reseroir outputs, produced by the given rtesting
        study_dict - dict specifying the new hyperparameter setting of the ESN 
     '''

    esn, config_istudy, study_tuple, istudy, nstudy, recompute_Win, recompute_Wres, recompute_Wfb = thread_args
    study_dict = esn.SetStudyParameters(config_istudy, study_tuple)

    loss_dict, y_pred_test, y_pred_val = run_turbESN(esn,
                                                    recompute_Win=recompute_Win,
                                                    recompute_Wres=recompute_Wres,
                                                    recompute_Wfb=recompute_Wfb)

    if esn.id % _ID_PRINT == 0: 
        logger.warn("ID {0}: study {1}/{2} done.".format(esn.id, istudy+1, nstudy))

    return (istudy, loss_dict, y_pred_test, y_pred_val, study_dict)
#-------------------------------------------------------------------------------------------------------- 
def parallelize_seeds(launch_thread_args) -> Tuple[int, list, int, str]:
    ''' Launches ThreadPool with maximum of MAX_THREADS threads Each thread runs an ESN setting (same RNG seed). 
        The ThreadPool is finished when all studies have been run. 
    
    INPUT:
        esn           - ESN object 
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to 
        MAX_THREADS   - max. no. threads per process
        config        - study configuration
        nsetting      - no. ESN grid search settings
        study_tuple   - tuple specifying the study parameters
    RETURN:
        esn_id        - ESN id
        randomSeed    - RNG seed of the ESN runs
        study_results - results of all studies (same RNG seed). Final shape: [RESULTS RUN1, ..., RESULTS RUN nsetting], 
                        where all RESULTS has shape (istudy, mse_train, mse_test, y_pred, study_dict)
        nsetting      - no. ESN grid search settings
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to 
    '''

    esn, filepath_esn, MAX_THREADS, config, nsetting, study_tuple = launch_thread_args

    study_results = []
    logger.warn('Subprocess {0} starting.'.format(esn.id))
    
    #----------------------------------
    #Set Seed
    #----------------------------------
    torch.manual_seed(esn.randomSeed)             

    #------------------------------------------
    #Assess whether weights must be recomputed
    #------------------------------------------
    recompute_Win  = False
    recompute_Wres = False
    recompute_Wfb  = False

    for param in study_tuple:
        if HP_dict[param]["CHANGES_Win"]:
            recompute_Win = True
        if HP_dict[param]["CHANGES_Wres"]:
            recompute_Wres = True
        if HP_dict[param]["CHANGES_Wfb"]:
            recompute_Wfb = True
        
    if not recompute_Win:
        esn.createInputMatrix()
    if not recompute_Wres:
        esn.createReservoirMatrix()
    if not recompute_Wfb:
        esn.createFeebackMatrix()

    #----------------------------------
    #Launch ThreadPool
    #----------------------------------
    executor = futures.ThreadPoolExecutor(MAX_THREADS)

    with executor as ex:
        study_counter = 0
        while study_counter < nsetting:
            
            num_threads = MAX_THREADS
            if (nsetting-study_counter) < MAX_THREADS:
                num_threads  = nsetting-study_counter

            # execute threads
            thread_results = ex.map(thread_run_turbESN, ([deepcopy(esn), config[isetting], study_tuple, isetting, nsetting, recompute_Win,recompute_Wres,recompute_Wfb] 
                                                 for ii,isetting 
                                                 in enumerate(np.arange(study_counter,study_counter+num_threads)))
                                    )
            study_results.append(list(thread_results))
            study_counter += num_threads

   
    logger.warn('ID {0} all done.'.format(esn.id))
    
    #----------------------------------
    #Reshape result list
    #----------------------------------
    
    study_results = [result for chunks in study_results for result in chunks]   

    return (esn.id, esn.randomSeed, study_results, nsetting, filepath_esn)
#--------------------------------------------------------------------------------------------------------
def callback_seeds(callback_args):
    ''' Callback to main process. Saves results of each seed into open hdf5 file f.
        ESN ID is the identifier for the ESN RNG seed. 
    INPUT:
        esn_id        - ESN id
        randomSeed    - RNG seed of the ESN runs
        study_results - results of all studies (same RNG seed). Final shape: [RESULTS RUN1, ..., RESULTS RUN nsetting], 
                        where all RESULTS have shape (istudy, mse_train, mse_test, y_pred, study_dict)
        nsetting        - no. studies
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to
    '''
    
    esn_id, randomSeed, study_results, nsetting, filepath_esn = callback_args

    if esn_id % _ID_PRINT == 0:
        logger.debug('ID {0}: returning to main process for saving.'.format(esn_id))

    for ii in range(nsetting):
        isetting, loss_dict, y_pred_test, y_pred_val, study_dict= study_results[ii] 

        save_study(filepath=filepath_esn, 
                   randomSeed=randomSeed, 
                   isetting=isetting,  
                   study_dict=study_dict, 
                   y_pred_test=y_pred_test, 
                   y_pred_val=y_pred_val, 
                   loss_dict=loss_dict) 
        
    if esn_id % _ID_PRINT == 0:        
        logger.warn('Saved study for ID {0}.'.format(esn_id))

#--------------------------------------------------------------------------------------------------------
def parallelize_settings(launch_thread_args) -> Tuple[int, list, int, str]:
    ''' Launches ThreadPool with maximum of MAX_THREADS threads Each thread runs the same ESN setting, but different RNG seed. 
        The ThreadPool is finished when all studies have been run. 
    
    INPUT:
        esn           - ESN object 
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to 
        MAX_THREADS   - max. no. threads per process
        config_study  - configuration of this particular study
        isetting      - study identifier
        seeds         - list of RNG seeds 
        study_tuple   - tuple specifying the study parameters
    RETURN:
        esn_id        - ESN id
        study_results - results of all studies (same ESN grid search setting). Final shape: [RESULTS RUN1, ..., RESULTS RUN nseed], 
                        where all RESULTS has shape (isetting, mse_train, mse_test, y_pred, study_dict)
        seeds         - list of RNG seeds 
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to 
    '''

    esn, filepath_esn, MAX_THREADS, config_isetting, isetting, seeds, study_tuple = launch_thread_args
 
    nseed = len(seeds)
    study_results = []
    logger.warn('Subprocess {0} starting.'.format(esn.id))        

    #-----------------------------------------------
    # Weights must be recomputed for differnt seeds
    #-----------------------------------------------
    recompute_Win  = True
    recompute_Wres = True
    recompute_Wfb  = True

    #----------------------------------
    #Launch ThreadPool
    #----------------------------------
    executor = futures.ThreadPoolExecutor(MAX_THREADS)

    with executor as ex:
        study_counter = 0
        while study_counter < nseed:
            
            num_threads = MAX_THREADS
            if (nseed-study_counter) < MAX_THREADS:
                num_threads  = nseed-study_counter
                
            # assign ESN random seeds 
            esn_copies = [deepcopy(esn) for _ in range(num_threads)]
            for ii,esn_copy in enumerate(esn_copies):
                esn_copy.SetRandomSeed(seeds[study_counter+ii]) 

            # execute threads
            thread_results = ex.map(thread_run_turbESN, 
                                    ([esn_copies[ii],config_isetting,study_tuple,iseed,nseed,recompute_Win,recompute_Wres,recompute_Wfb] 
                                    for ii,iseed in enumerate(np.arange(study_counter,study_counter+num_threads)))
                                    )
            study_results.append(list(thread_results))
            study_counter += num_threads

   
    logger.warn('ID {0} all done.'.format(esn.id))
    
    #----------------------------------
    #Reshape result list
    #----------------------------------
    study_results = [result for chunks in study_results for result in chunks]   

    return (esn.id, study_results, seeds, filepath_esn)

#--------------------------------------------------------------------------------------------------------
def callback_settings(callback_args):
    ''' Callback to main process. Saves results of each seed into open hdf5 file f.
        ESN ID is the identifier for the ESN grid search setting. 
    
    INPUT:
        esn_id        - ESN id
        study_results - results of all studies (same RNG seed). Final shape: [RESULTS RUN1, ..., RESULTS RUN nseed], 
                        where all RESULTS have shape (iseed, mse_train, mse_test, y_pred, study_dict)
        seeds         - list of RNG seeds 
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to
    '''
    
    esn_id, study_results, seeds, filepath_esn = callback_args


    if esn_id % _ID_PRINT == 0:
        logger.debug('ID {0}: returning to main process for saving.'.format(esn_id))

    for ii,randomSeed in enumerate(seeds):
        iseed, loss_dict, y_pred_test, y_pred_val, study_dict= study_results[ii] 

        save_study(filepath=filepath_esn, 
                   randomSeed=randomSeed, 
                   isetting=esn_id,
                   study_dict=study_dict, 
                   y_pred_test=y_pred_test, 
                   y_pred_val=y_pred_val, 
                   loss_dict=loss_dict) 
        
    if esn_id % _ID_PRINT == 0:        
        logger.warn('Saved study for ID {0}.'.format(esn_id))

#--------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
# CROSS VALIDATION STUDY  (k-fold forward walk validaiton scheme, see Lukosevicius et al. (2021))
#-------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
def launch_thread_forward_validate_turbESN(thread_args):
    ''' Cross validation of given esn (see Lukosevicius et al. 2021).
    INPUT:
        esn    - ESN class object containing the reservoir parameters and training & testing data sets.
        cv     - CrossValidation class object  containing cross validation parameters and data
        config_istudy - study parameter configuration for this ESN run
        study_tuple   - set containing the names of the parameter that are studied
        istudy        - study ID

    RETURN:
        istudy     - Study ID
        mse_train  - mean square error of (teacher forced!) reservoir output (to training target data set y_train) for each fold. Mean w.r.t. timestep- & mode-axis. 
        mse_test   - mean square error of reservoir output (to testing data set y_test) for each fold. Mean w.r.t. timestep- & mode-axis. 
        mse_val    - mean square error of reservoir output (to validation data set y_val) for each fold. Mean w.r.t. timestep- & mode-axis. 
        y_pred_test- testing reseroir outputs, produced by the given reservoir specified in esn.
        y_pred_val - validation reseroir outputs, produced by the given reservoir specified in esn.       
        study_dict - dict specifying the new hyperparameter setting of the ESN 
         
     '''

    esn, cv, config_istudy, study_tuple, istudy, nstudy = thread_args
    torch.manual_seed(esn.randomSeed)

    study_dict = esn.SetStudyParameters(config_istudy, study_tuple)
    mse_train, mse_test, mse_val, y_pred_test, y_pred_val = forward_validate_auto_ESN(esn, cv)

    if esn.id % _ID_PRINT == 0: 
        logger.warn("ID {0}: study {1}/{2} done.".format(esn.id, istudy+1, nstudy))

    return (istudy, mse_train.numpy(), mse_test.numpy(), mse_val.numpy(), y_pred_test.numpy(), y_pred_val.numpy(), study_dict)
#--------------------------------------------------------------------------------------------------------

def launch_process_forward_validate_turbESN(launch_thread_args) -> Tuple[int, list, int, str]:
    ''' Launches ThreadPool with maximum of MAX_THREADS threads Each thread runs an ESN setting (same RNG seed). 
        The ThreadPool is finished when all studies have been run. 
    
    INPUT:
        esn           - ESN object 
        cv     - CrossValidation class object  containing cross validation parameters and data
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

    esn, cv, filepath_esn, MAX_THREADS, config, nstudy, study_tuple = launch_thread_args

    study_results = []
    logger.warn('Subprocess {0} starting.'.format(esn.id))
    
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

            thread_results = ex.map(launch_thread_forward_validate_turbESN, ([deepcopy(esn), deepcopy(cv), config[istudy], study_tuple, istudy, nstudy] 
                                                 for istudy 
                                                 in np.arange(study_counter,study_counter+num_threads))
                                    )
            study_results.append(list(thread_results))
            study_counter += num_threads

   
    logger.warn('ID {0} all done.'.format(esn.id))
    
    #----------------------------------
    #Reshape result list
    #----------------------------------
    
    study_results = [result for chunks in study_results for result in chunks]   

    return (esn.id, study_results, nstudy, filepath_esn)
#--------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
# EXPERIMENTAL
#-------------------------------------------------------------------------------------------------
def callback_seeds_postprocess(callback_args):
    ''' Callback to main process. Use ESN predictions for post-processing.
        Saves all results of each seed into open hdf5 file f.
        ESN ID is the identifier for the ESN RNG seed. 
    INPUT:
        esn_id        - ESN id
        randomSeed    - RNG seed of the ESN runs
        study_results - results of all studies (same RNG seed). Final shape: [RESULTS RUN1, ..., RESULTS RUN nsetting], 
                        where all RESULTS have shape (istudy, mse_train, mse_test, y_pred, study_dict)
        nsetting        - no. studies
        filepath_esn  - path to which the hdf5 file of the ESN study was saved to
    '''
    
    esn_id, randomSeed, study_results, nsetting, filepath_esn = callback_args

    if esn_id % _ID_PRINT == 0:
        logger.debug('ID {0}: returning to main process for post-processing & saving.'.format(esn_id))

    for ii in range(nsetting):
        isetting, mse_train, mse_test, mse_val, y_pred_test, y_pred_val, study_dict= study_results[ii] 

        # TO DO
        # add post processing
        # fields_true, fields_pred = reconstruct_form_pod(y_true,y_pred_test,spatial_modes)
        # lta_true, lta_pred = compute_LTA()
        # nare = compute_NARE(lta_true,lta_pred)
        # error_dict = {'nare': nare}
        save_study(filepath=filepath_esn, iseed=esn_id, isetting=isetting,  
                  study_dict=study_dict, y_pred_test=y_pred_test, y_pred_val=y_pred_val, 
                  mse_train=mse_train, mse_test=mse_test, mse_val=mse_val,randomSeed=randomSeed) #error_dict
        
    if esn_id % _ID_PRINT == 0:        
        logger.warn('Saved study for ID {0}.'.format(esn_id))

#--------------------------------------------------------------------------------------------------------