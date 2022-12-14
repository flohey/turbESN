###############################################################################################
# This script reads pre-computed grid/random searches
###############################################################################################
# DEVICE: CPU
###############################################################################################
# Florian Heyder - 14.11.2022
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
from turbESN.util import (undo_minmax_scaling,)
from turbESN.core import *
from turbESN.util import read_study


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
    print("Reading yaml-config file.")

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

    # Path config
    #-------------------
    path_data =  yaml_config["path_data"]                      
    filename_data = yaml_config["filename_data"]
    path_esn = yaml_config["path_esn"]                       
    subdir_esn = yaml_config["subdir_esn"]

    #----------------------------------------------------
    # 2. PATH PARAMETERS
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
    filepath_esn = os.path.join(path_esn,filename_esn)

    assert os.path.exists(filepath_esn), f'Error: ESN filepath not found! (filepath_esn)'
    
    #----------------------------------------------------
    # 3. READ STUDY
    #----------------------------------------------------
    print(f'Reading grid search file {filepath_esn}')
    time_start = time.time()
    esn = ESN.read(filepath_esn)
        
    MSE_TRAIN, MSE_TEST, MSE_VAL, Y_PRED_TEST, Y_PRED_VAL = read_study(filepath_esn, 
                                                                       study_tuple=study_tuple,
                                                                       read_pred=True, 
                                                                       iseeds=range(nseed))
    time_end = time.time()
    print('\n Elapsed time {0:.2f}s'.format(time_end-time_start))
    #----------------------------------------------------
    # 3. READ STUDY
    #----------------------------------------------------

    with h5py.File(filepath_esn,'a') as f:
        #x_min_in = torch.tensor(f["Data/x_min_in"])
        #x_max_in = torch.tensor(f["Data/x_max_in"])
        x_min_out = torch.tensor(f["Data/x_min_out"])
        x_max_out = torch.tensor(f["Data/x_max_out"])

        for iseed in range(nseed):
            esn_id = iseed
            y_pred_test = Y_PRED_TEST[iseed]
            y_pred_test = [undo_minmax_scaling(y_pred, x_min_out, x_max_out) for y_pred in y_pred_test]
            y_test = undo_minmax_scaling(esn.y_test, x_min_out, x_max_out) 

            pass
            #method w. args: func(esn_id, y_pred_test, y_test, rest needed for reconstruction)

    time_end = time.time()
    print('\n ----------------------------------------')
    print('\n PROGRAM FINISHED!')
    print('\n ----------------------------------------')
    print('\n Total elapsed time {0:.2f}s'.format(time_end-time_start))