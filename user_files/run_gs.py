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
    compute_nrmse
    )
from turbESN.core import *
from turbESN.study import parallelize_seeds, parallelize_settings, callback_seeds, callback_settings


logging.warning("Using: turbESN v."+turbESN.__version__)

# Load YAML config file
#---------------------------
with open("esn_config.yaml","r") as f:
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

    # case params
    #----------------------------


    # ESN (general)
    #-------------------
    n_reservoir = yaml_config["n_reservoir"]
    leakingRate = yaml_config["leakingRate"]
    spectralRadius =  yaml_config["spectralRadius"]
    reservoirDensity = yaml_config["reservoirDensity"]
    regressionParameter = yaml_config["regressionParameter"]
    mode = yaml_config["mode"]
    verbose = yaml_config["verbose"]

    bias_in = yaml_config["bias_in"]
    bias_out = yaml_config["bias_in"]
    outputInputScaling = yaml_config["outputInputScaling"]
    inputScaling = yaml_config["inputScaling"]
    feedbackScaling = yaml_config["feedbackScaling"]
    inputDensity = yaml_config["inputDensity"]
    noiseLevel_in = yaml_config["noiseLevel_in"]
    noiseLevel_out = yaml_config["noiseLevel_out"]
    extendedStateStyle = yaml_config["extendedStateStyle"]
    use_feedback = yaml_config["use_feedback"]
    weightGeneration = yaml_config["weightGeneration"]
    fit_method = yaml_config["fit_method"]

    # ESN (data layout)
    #-------------------
    dataScaling = yaml_config["dataScaling"]
    trainingLength = yaml_config["trainingLength"]
    testingLength = yaml_config["testingLength"]
    validationLength = yaml_config["validationLength"]
    transientTime = yaml_config["transientTime"]
    esn_start = yaml_config["esn_start"]

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
    data_str =  yaml_config["data_str"]
    if data_str != "":
        data_str += "_"

    study_str = ""
    for param_str in study_tuple:
        study_str += param_str +"_"

    N_str = f"N{n_reservoir}_"
    D_str = "D{0:.1}_".format(reservoirDensity).replace('.','')
    regParam_str = "regParam{0:.1e}_".format(regressionParameter).replace('-','').replace('+','').replace('.','').replace('0','')
    SR_str = "SR{0:.2}_".format(spectralRadius).replace('.','')
    LR_str = "LR{0:.2}_".format(leakingRate).replace('.','')
    dS_str = f"dS{dataScaling}_"
    esn_start_str = f"esn_start{esn_start}_"
    TL_str = "TL{0:.1e}".format(trainingLength).replace('-','').replace('+','').replace('.','').replace('0','')

    filename_esn = data_str + study_str + N_str + D_str + SR_str + LR_str + regParam_str + dS_str + esn_start_str + TL_str + '.hdf5'
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
        data = np.array(f["time_coefficients"])

    data_timesteps, n_input_data= data.shape
    
    n_input = yaml_config["n_input"]
    if mode == "auto":
        n_output = n_input
        data_in = data[:,:n_input]
    elif mode == "teacher":
        try:
            n_output = yaml_config["n_output"]
        except KeyError:
            logging.fatal("In teacher mode: specify n_input & n_output in yaml config.")
            exit()

    data_in = data[:,:n_input]
    data_out = data[:,:n_output]
    
    print(f"Using ESN in mode {mode}")
    print(f"n_input = {n_input}, n_output = {n_output}\n")
    # Data parameters
    #------------------
    esn_end   = esn_start + (trainingLength+testingLength+validationLength)
    assert esn_end <= data_timesteps, f"esn_end ({esn_end}) must be <= available data timesteps ({data_timesteps})"

    #normalize data to [-dataScaling,dataScaling] (along time-axis in training phase)
    x_min_in = np.min(data_in[esn_start:esn_start+trainingLength],axis=0)
    x_max_in = np.max(data_in[esn_start:esn_start+trainingLength],axis=0)
    data_in_scaled = minmax_scaling(data_in,x_min=x_min_in,x_max=x_max_in,dataScaling=dataScaling) 

    x_min_out = np.min(data_out[esn_start:esn_start+trainingLength],axis=0)
    x_max_out = np.max(data_out[esn_start:esn_start+trainingLength],axis=0)
    data_out_scaled = minmax_scaling(data_out,x_min=x_min_out,x_max=x_max_out,dataScaling=dataScaling) 

    #----------------------------------------------------
    # 5. ESN PARAMETERS
    # - create ESN
    # - fix ESN HP
    # - set ESN training and testing data
    #----------------------------------------------------
    esn = ESN(randomSeed = seeds,
            esn_start = esn_start,
            esn_end = esn_end,
            trainingLength=trainingLength,
            testingLength=testingLength,
            validationLength=validationLength,
            data_timesteps = data_timesteps,
            n_input = n_input,
            n_output = n_output,
            n_reservoir = n_reservoir,
            leakingRate = leakingRate,
            spectralRadius = spectralRadius,
            reservoirDensity = reservoirDensity,
            regressionParameter = regressionParameter,
            bias_in = bias_in,
            bias_out = bias_out,
            outputInputScaling = outputInputScaling,
            inputScaling = inputScaling,
            inputDensity = inputDensity,
            noiseLevel_in = noiseLevel_in,
            noiseLevel_out = noiseLevel_out,
            mode = mode,
            weightGeneration = weightGeneration,
            extendedStateStyle = extendedStateStyle,
            transientTime  = transientTime, 
            use_feedback=use_feedback,
            feedbackScaling=feedbackScaling,
            fit_method=fit_method,
            verbose = verbose)


    if mode == "auto":
        u_train, y_train, u_test, y_test, u_val, y_val = prepare_auto_data( data=data_in_scaled,
                                                                            n_input=n_input, 
                                                                            trainingLength=trainingLength, 
                                                                            testingLength=testingLength, 
                                                                            esn_start=esn_start, 
                                                                            esn_end=esn_end,
                                                                            validationLength=validationLength)

    elif mode == "teacher":
        u_train, y_train, u_test, y_test, u_val, y_val = prepare_teacher_data(  data_in=data_in_scaled,
                                                                                data_out=data_out_scaled,
                                                                                n_input=n_input, 
                                                                                n_output=n_output,
                                                                                trainingLength=trainingLength, 
                                                                                testingLength=testingLength, 
                                                                                esn_start=esn_start, 
                                                                                esn_end=esn_end,
                                                                                validationLength=validationLength)
   
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
                pool.apply_async(parallelize_settings, args=((esn_copy, filepath_esn, MAX_THREADS, setting, isetting, seeds, study_tuple),), callback=callback_settings(callback_args))
            
        pool.close()
        pool.join()
    time_end = time.time()
    print('----------------------------------------\n')
    print('PROGRAM FINISHED!')
    print('(total elapsed time {0:.2f}min)\n'.format((time_end-time_start)/60))
    print('----------------------------------------')

#---------------------------------------------------------------------------------------------
#                          END OF PROGRAM
#---------------------------------------------------------------------------------------------
   
