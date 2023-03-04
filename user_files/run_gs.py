###############################################################################################
# This script runs ensemble grid/random searches
###############################################################################################
# DEVICE: CPU
###############################################################################################

#Parallelization
import multiprocessing as mp

#Misc. 
import sys
import os
import time
from copy import deepcopy
import logging

#Backends
import numpy as np
import torch

#Data structure
import h5py
import yaml

#turbESN
import turbESN
from turbESN.util import (
    prepare_teacher_data,  
    prepare_auto_data, 
    create_hdf5_groups, 
    init_random_search,
    init_grid_search,
    minmax_scaling,
    compute_mse,
    compute_nrmse,
    compute_kld,
    get_file_name
    )
from turbESN.core import *
from turbESN.study import parallelize_seeds, parallelize_settings, callback_seeds, callback_settings


logging.warning("Using: turbESN v."+turbESN.__version__)

# Load YAML config file
#---------------------------
yaml_config_path = "esn_config.yaml"
with open(yaml_config_path,"r") as f:
    yaml_config = yaml.safe_load(f)
#---------------------------

MAX_PROC = yaml_config["MAX_PROC"]                  #no. subprocesses (here: each running a seed)
MAX_THREADS = yaml_config["MAX_THREADS"]            #max. no. threads per subprocess (if not using a full machine: set to 1)

if MAX_PROC is None:
    MAX_PROC = int(os.getenv("LSB_DJOB_NUMPROC")) 

if __name__ == '__main__':    

    #----------------------------------------------------
    #1. Load ESN Config 
    #----------------------------------------------------
    print("\nReading yaml-config file.\n")
    # ESN (general)
    #-------------------
    esn = ESN.read_yaml(yaml_config_path)

    # ESN (study)
    #-------------------
    nseed = yaml_config["nseed"]
    seeds = range(nseed)
                                            
    study_tuple = yaml_config["study_tuple"]  
    study_param_limits = yaml_config["study_param_limits"]
    study_param_list = yaml_config["study_param_list"]

    if study_param_list is None:
        study_param_list = [None for _ in study_tuple]
    
    assert len(study_param_list) == len(study_tuple), "Error: study_param_list must have same length as study_tuple!"
    do_random_search = yaml_config["do_random_search"]   
    use_parallel_setting = yaml_config["use_parallel_setting"]

    # Path config
    #-------------------
    dataScaling = yaml_config["dataScaling"]
    path_data =  yaml_config["path_data"]                      
    filename_data = yaml_config["filename_data"]
    path_esn = yaml_config["path_esn"]                       
    subdir_esn = yaml_config["subdir_esn"]

    #----------------------------------------------------
    # 2. PATH PARAMETERS
    # - specify where the study should be saved to
    # - specify where the data is located
    #----------------------------------------------------

    # Prepare filename
    #-------------------
    filename_esn = get_file_name(yaml_config_path)
    path_esn = os.path.join(path_esn,subdir_esn)
    os.makedirs(path_esn,exist_ok=True)
    filepath_esn = os.path.join(path_esn,filename_esn)

    #----------------------------------------------------
    # 3.INIT GRID/RANDOM SEARCH
    # - specify hyperparameters to study and their range
    #----------------------------------------------------
    #----------------------------------------------------        
    if not do_random_search:
        print("Initializing grid search...")
        config, nsetting  = init_grid_search(study_tuple, study_param_limits, study_param_list)
        print('Seeds: {0}. Studies per seed: {1}\n'.format(nseed, nsetting))
    else:
        print("Initializing random search...")
        nsetting = yaml_config["nsetting"]

        config, nsetting = init_random_search(nsetting, study_tuple,limits=study_param_limits)
        print('Seeds: {0}. Studies per seed: {1}\n'.format(nseed, nsetting))

    #----------------------------------------------------
    # 4. INIT ESN DATA
    # - import data 
    # - scale data
    #----------------------------------------------------
    print("Importing data")
    with h5py.File(path_data + filename_data,'r') as f:
        data = np.array(f["data"])

    esn.data_timesteps, n_input_data= data.shape
    
    if esn.mode == "auto":
        esn.n_output = esn.n_input

    data_in  = data[:,:esn.n_input]
    data_out = data[:,:esn.n_output]
    
    print(f"Using ESN in mode {esn.mode}")
    print(f"n_input = {esn.n_input}, n_output = {esn.n_output}\n")
    # Data parameters
    #------------------
    esn.esn_end   = esn.esn_start + (esn.trainingLength+esn.testingLength+esn.validationLength)
    assert esn.esn_end <= esn.data_timesteps, f"esn_end ({esn.esn_end}) must be <= available data timesteps ({esn.data_timesteps})"

    #normalize data to [-dataScaling,dataScaling] (along time-axis in training phase)
    x_min_in = np.min(data_in[esn.esn_start:esn.esn_start+esn.trainingLength],axis=0)
    x_max_in = np.max(data_in[esn.esn_start:esn.esn_start+esn.trainingLength],axis=0)
    data_in_scaled = minmax_scaling(data_in,x_min=x_min_in,x_max=x_max_in,dataScaling=dataScaling) 

    x_min_out = np.min(data_out[esn.esn_start:esn.esn_start+esn.trainingLength],axis=0)
    x_max_out = np.max(data_out[esn.esn_start:esn.esn_start+esn.trainingLength],axis=0)
    data_out_scaled = minmax_scaling(data_out,x_min=x_min_out,x_max=x_max_out,dataScaling=dataScaling) 

    #----------------------------------------------------
    # 5. Set ESN Data
    # - set ESN training and testing data
    #----------------------------------------------------

    if esn.mode == "auto":
        u_train, y_train, u_test, y_test, u_val, y_val = prepare_auto_data( data=data_in_scaled,
                                                                            n_input=esn.n_input, 
                                                                            trainingLength=esn.trainingLength, 
                                                                            testingLength=esn.testingLength, 
                                                                            esn_start=esn.esn_start, 
                                                                            esn_end=esn.esn_end,
                                                                            validationLength=esn.validationLength)

    elif esn.mode == "teacher":
        u_train, y_train, u_test, y_test, u_val, y_val = prepare_teacher_data(  data_in=data_in_scaled,
                                                                                data_out=data_out_scaled,
                                                                                n_input=esn.n_input, 
                                                                                n_output=esn.n_output,
                                                                                trainingLength=esn.trainingLength, 
                                                                                testingLength=esn.testingLength, 
                                                                                esn_start=esn.esn_start, 
                                                                                esn_end=esn.esn_end,
                                                                                validationLength=esn.validationLength)
   
    esn.loss_func=dict(mse=compute_mse,nrmse=compute_nrmse)
    
    esn.SetTrainingData(u_train=u_train, y_train=y_train)
    esn.SetTestingData(y_test=y_test, test_init_input= y_train[-1:,:], u_test = u_test)
    esn.SetValidationData(y_val=y_val, u_val = u_val, val_init_input=y_test[-1:,:],u_pre_val=u_test) 

    #----------------------------------------------------
    # 6. RUN STUDY 
    # - copy ESN parameters to each subprocess
    # - distribute different seeds among processes
    # - distribute different ESN settings among threads
    #----------------------------------------------------
    esn.to_torch()
    if not os.path.exists(filepath_esn):
        esn.save(filepath_esn)
        
    create_hdf5_groups(filepath_esn, seeds, nsetting)

    with h5py.File(filepath_esn,'a') as f:
        if 'x_min_in' not in f['Data']:
            f["Data"].create_dataset(name='x_min_in',data=x_min_in,compression='gzip',compression_opts=9)
        if 'x_max_in' not in f['Data']:
            f["Data"].create_dataset(name='x_max_in',data=x_max_in,compression='gzip',compression_opts=9)
        if 'x_min_out' not in f['Data']:
            f["Data"].create_dataset(name='x_min_out',data=x_min_out,compression='gzip',compression_opts=9)
        if 'x_max_out' not in f['Data']:
            f["Data"].create_dataset(name='x_max_out',data=x_max_out,compression='gzip',compression_opts=9)

    # Check echo state property
    esn.createWeightMatrices()
    estimated_transientTime = esn.verify_echo_state_property(u=u_train,y=y_train)
    print("Estimated transientTime = {0}\n".format(estimated_transientTime))

    # Print ESN info
    print(esn)

    # Start study
    time_start = time.time()
    pool = mp.Pool(processes=MAX_PROC)

    with h5py.File(filepath_esn,'a') as f: 
    
        # parallelize RNG seed
        if not use_parallel_setting:
            print('Distributing seeds over processes\n')
            for esn_id,seed in enumerate(seeds):
                esn_copy = deepcopy(esn)
                esn_copy.SetRandomSeed(seed)
                esn_copy.SetID(esn_id)
                pool.apply_async(parallelize_seeds, args=((esn_copy, filepath_esn, MAX_THREADS, config, nsetting, study_tuple),), callback=callback_seeds)       
        else:
            print('Distributing settings over processes\n')
            for esn_id,isetting in enumerate(range(nsetting)):
                esn_copy = deepcopy(esn)
                setting = config[isetting]
                esn_copy.SetID(esn_id)
                pool.apply_async(parallelize_settings, args=((esn_copy, filepath_esn, MAX_THREADS, setting, isetting, seeds, study_tuple),), callback=callback_settings)
            
        pool.close()
        pool.join()
    time_end = time.time()
    print('----------------------------------------')
    print('PROGRAM FINISHED (elapsed time {0:.2f}min)'.format((time_end-time_start)/60))
    print('----------------------------------------')

#---------------------------------------------------------------------------------------------
#                          END OF PROGRAM
#---------------------------------------------------------------------------------------------
   
