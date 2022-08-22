###############################################################################################
# This script runs ensemble grid/random searches
###############################################################################################
# DEVICE: CPU
###############################################################################################
# Florian Heyder - 17.05.2022
###############################################################################################

#Parallelization
import multiprocessing as mp
from concurrent import futures

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
    PrepareTeacherData,  
    PreparePredictorData, 
    InitStudyOrder,
    CreateHDF5Groups, 
    InitRandomSearchStudyOrder,
    minmax_scaling
    )
from turbESN.core import *
from turbESN.study import launch_thread_RunturbESN, launch_process_RunturbESN, Callback


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
    #Load ESN Config 
    #----------------------------------------------------
    print("Reading yaml-config file.")

    # case params
    #----------------------------


    # ESN config
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

    # ESN Data config
    #-------------------
    dataScaling = yaml_config["dataScaling"]
    trainingLength = yaml_config["trainingLength"]
    testingLength = yaml_config["testingLength"]
    validationLength = yaml_config["validationLength"]
    transientTime = yaml_config["transientTime"]
    
    # ESN Study config
    #-------------------
    nseed = yaml_config["nseed"]
    randomSeed = range(nseed)                                              
    esn_ids = range(nseed) 

    doRandomSearch = yaml_config["doRandomSearch"]                                               
    study_tuple = yaml_config["study_tuple"]  
    study_param_limits = yaml_config["study_param_limits"]
    study_param_list = yaml_config["study_param_list"]

    if study_param_list is None:
        study_param_list = [None for _ in study_tuple]
    
    assert len(study_param_list) == len(study_tuple), "Error: study_param_list must have same length as study_tuple!"

    # Path config
    #-------------------
    path_data =  yaml_config["path_data"]                      
    filename_data = yaml_config["filename_data"]
    path_esn = yaml_config["path_esn"]                       
    comment_str = yaml_config["comment_str"]

    #----------------------------------------------------
    #PATH PARAMETERS
    # - specify where the study should be saved to
    # - specify where the data is located
    #----------------------------------------------------

    # Prepare filename
    #-------------------
    data_str = f''

    study_str = ""
    for param_str in study_tuple:
        study_str += "_"+param_str 

    N_str = f"_N{n_reservoir}"
    D_str = "_D{0:.1}".format(reservoirDensity).replace('.','')
    regParam_str = "_regParam{0:.1e}".format(regressionParameter).replace('-','').replace('+','').replace('.','').replace('0','')
    dS_str = f"_dS{dataScaling}"
    TL_str = "_TL{0:.1e}".format(trainingLength).replace('-','').replace('+','').replace('.','').replace('0','')

    # Create filename
    #-------------------
    filename_esn = filename_data[:-5] + data_str + study_str + N_str + D_str + regParam_str + dS_str + TL_str  + comment_str + '.hdf5'
    filepath_esn = path_esn + filename_esn

    #----------------------------------------------------
    #STUDY PARAMETERS
    # - choose grid or random search mode
    # - specify hyperparameters to study and their range
    #----------------------------------------------------
    #----------------------------------------------------        

    if doRandomSearch:
        nstudy = yaml_config["nstudy"]
        use_log_scale = yaml_config["use_log_scale"]

        if len(study_tuple) == len(study_param_limits):                #only use if correctly specified
            HP_range_dict = dict(zip(study_tuple,study_param_limits))
        
        config = InitRandomSearchStudyOrder(nstudy, study_tuple,HP_range_dict=HP_range_dict, use_log_scale=use_log_scale)
        print('Random Search. Seeds: {0}. Studies per seed: {1}\n'.format(nseed, nstudy))
    else:
        study_parameters = []
        for ii,limits in enumerate(study_param_limits): 
            
            # HP grid values are read from config: study_param_list
            if len(study_param_list[ii]) is not None and len(study_param_list[ii]) != 0:
                param_val = study_param_list[ii]

            # HP grid is computed from limits in config: study_param_limits
            else:
                assert len(limits) == 4, "Error: study_param_limits must be of style [value_min, value_max, nvals, use_log]!"
                x0,x1,nx, use_log = limits

                if use_log:
                    param_val = np.logspace(x0,x1,nx)
                else:
                    param_val = np.linspace(x0,x1,nx)

                # leakingRate specified for each neuron: take param_val as leakingRate_min values
                if study_tuple[ii] == "leakingRate_neural":
                    param_val = [loguniform.rvs(lr_min-1e-9,1,size=(n_reservoir,1)) for lr_min in param_val]

            study_parameters.append(param_val)   

        study_parameters = tuple(study_parameters)

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
        data = np.array(f.get('POD/time_coefficients')).real

    data_timesteps, n_input_data = data.shape

    # Data parameters
    #------------------
    esn_start = data_timesteps - (trainingLength+testingLength+validationLength)
    esn_end = data_timesteps 
    n_input = n_input_data
    n_output = n_input_data

    #normalize data to [-dataScaling,dataScaling] (along time-axis in training phase)
    x_min = np.min(data[esn_start:esn_start+trainingLength],axis=0)
    x_max = np.max(data[esn_start:esn_start+trainingLength],axis=0)
    data_scaled = minmax_scaling(data, x_min=x_min, x_max=x_max, dataScaling=dataScaling)

    #----------------------------------------------------
    #ESN PARAMETERS
    # - create ESN
    # - adapt ESN setting
    # - adapt ESN training and testing data
    #----------------------------------------------------
    esn = ESN(randomSeed = randomSeed,
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
            verbose = verbose)


    u_train, y_train, u_test, y_test, u_val, y_val = PreparePredictorData(  data=data_scaled,
                                                                            n_input=n_input, 
                                                                            trainingLength=trainingLength, 
                                                                            testingLength=testingLength, 
                                                                            esn_start=esn_start, 
                                                                            esn_end=esn_end,
                                                                            validationLength=validationLength)
                                                            
    esn.SetTrainingData(u_train=u_train, y_train=y_train)
    esn.SetTestingData(y_test=y_test, pred_init_input= y_train[-1:,:], u_test = u_test)
    esn.SetValidationData(y_val=y_val, u_val = u_val, val_init_input=y_test[-1:,:]) 


    #----------------------------------------------------
    #RUN STUDY 
    # - copy ESN parameters
    # - change seed
    # - distribute different seeds among processes
    # - distribute different ESN settings among threads
    #----------------------------------------------------
    esn.toTorch()
    esn.save(filepath_esn)
    CreateHDF5Groups(filepath_esn, esn_ids, nstudy)

    print(esn)

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
   
