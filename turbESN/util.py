#backends
import numpy as np
import torch

#save data structure
import h5py 

#turbESN
from .core import (ESN, _DTYPE, _DEVICE, _ESN_MODES, _WEIGTH_GENERATION, _LOGGING_FORMAT)

#misc
import sys
import os
from copy import deepcopy
import logging
from typing import Union, Tuple, List

from scipy.stats import wasserstein_distance
from scipy.stats import loguniform


_MSE_DEFAULT = 1e6  #default value for the mean square error, if error occurs
###########################################################################################################

#                             PRE-PROCESSING/ IMPORTING

###########################################################################################################
#--------------------------------------------------------------------------
def PreparePredictorData(data: Union[np.ndarray, torch.Tensor], 
                         n_input: int, 
                         trainingLength: int, 
                         testingLength: int, 
                         esn_start: int, 
                         esn_end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' Prepares the input and output training and testing/validation data set for the autonomous predictor ESN case, i.e. where in the
        testing pase, the output data is fed back into the reservoir input layer.
        The data set u_test is therefore not needed, as there is only one initial input in the testing phase.
        Note that n_output = n_input. 

    INPUT:
        data           - data array which is supposed to be used for the ESN. The shape must be (data_timesteps, nmodes)
        n_input        - input dimensions/ no. input modes
        trainingLength - no. time steps for the training data set
        testingLength  - no. time steps for the testing/validation data set
        esn_start      - Index of the original data, at which the training output y_train will begin. 
        esn_end        - Index of the original data, at which the testing/validation output y_test will end.

    RETURN:
        u_train - training input data set. Shape: (trainingLength, n_input)
        y_train - training output data set. Shape: (trainingLength, n_output)
        u_test  - testing/validation input data set (not used in auotnomous predictor mode). Shape: (testingLength, n_input)
        y_test  - testing/validation output data set. Shape: (testingLength, n_output)
    '''
    
    data_esn = data[esn_start-1:esn_end,:n_input]
    u_train = data_esn[0:trainingLength,:]
    y_train = data_esn[1:trainingLength+1,:]
    
    u_test = data_esn[trainingLength:trainingLength+testingLength,:]
    y_test = data_esn[trainingLength+1:,:]

    return torch.as_tensor(u_train, dtype = _DTYPE), torch.as_tensor(y_train, dtype = _DTYPE),torch.as_tensor(u_test, dtype = _DTYPE),torch.as_tensor(y_test, dtype = _DTYPE)
#--------------------------------------------------------------------------
def PrepareTeacherData(data_in: Union[np.ndarray, torch.Tensor], 
                       data_out: int, 
                       n_input: int, 
                       n_output: int, 
                       trainingLength: int, 
                       testingLength: int, 
                       esn_start: int, 
                       esn_end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' Prepares the input and output training and testing/validation data set for the teacher forced ESN case, i.e. where the
        reservoir input is a teacher signal. 

    INPUT:
        data_in        - data array to be used for the ESN. The shape must be (data_timesteps, n_input)
        data_out       - data array to be used for the ESN. The shape must be (data_timesteps, n_output)
        n_input        - input dimensions/ no. input modes
        n_output       - output dimensions/ no. output modes
        trainingLength - no. time steps for the training data set
        testingLength  - no. time steps for the testing/validation data set
        esn_start      - Index of the original data, at which the training output y_train will begin. 
        esn_end        - Index of the original data, at which the testing/validation output y_test will end.

    RETURN:
        u_train - training input data set. Shape: (trainingLength, n_input)
        y_train - training output data set. Shape: (trainingLength, n_output)
        u_test  - testing/validation input data set.  Shape: (testingLength, n_input)
        y_test  - testing/validation output data set. Shape: (testingLength, n_output)
    '''
    data_in_esn = data_in[esn_start-1:esn_end,:n_input]
    data_out_esn = data_out[esn_start-1:esn_end,:n_output]

    u_train = data_in_esn[0:trainingLength]
    y_train = data_out_esn[0:trainingLength]
    
    u_test = data_in_esn[trainingLength:trainingLength+testingLength]
    y_test = data_out_esn[trainingLength:trainingLength+testingLength]

    return torch.as_tensor(u_train, dtype = _DTYPE), torch.as_tensor(y_train, dtype = _DTYPE),torch.as_tensor(u_test, dtype = _DTYPE),torch.as_tensor(y_test, dtype = _DTYPE)

#--------------------------------------------------------------------------
def Recursion(iparam: int, iterators: np.ndarray, study_parameters: tuple):
    ''' Iterates the iterators which are used to change the hyperparmeters. 
        Makes sure that all combinations of the parameters in study_parameters are used.

        INPUT:
            iparam           - index which defines the hyperparameter that is changed
            iterators        - iterator that defines the no. ESNs that have been run with changing one hyperparameter (study_parameters[iparam]) 
            study_parameters - tuple specifying the range of the parameters that are studies

    FH 22.03.21: moved function from esn_user_mp.py/esn_user_thread.py to this file
    '''
    
    #if iterator associated with iparam hasn't reached final value --> increment
    if iterators[iparam] < len(study_parameters[iparam])-1:
        iterators[iparam] +=1
            
    #if iterator associated with iparam has reached final value --> reset iterator and increment/reset higher level iterator associated with iparam-1.
    else:
        iterators[iparam] = 0
        Recursion(iparam-1,iterators,study_parameters)

#--------------------------------------------------------------------------
def InitStudyOrder(nstudy: int, study_parameters: tuple) -> list:
    ''' Initializes the study parameter settings into an array.

    INPUT:
        nstudy           - number of different reservoir settings that were studied. If nstudy = None, the number is deduced from the file.
        study_parameters - tuple specifying the range of the parameters that are studied


    RETURN: 
        config - list indicating the parameter setting for given study
    ''' 
    
    assert len(study_parameters) != 0,'study_parameters are empty. Did you forget to specify the range of the studied HP?'
    assert nstudy > 0, 'nstudy ({0}) must be > 0'.format(nstudy)

    config = []
    nstudyparameters = len(study_parameters)
    iterators = np.zeros([nstudyparameters], dtype=int)

    for itotal in range(nstudy):
        #Update iterators:      
        if itotal == 0:
            pass
        else:
            Recursion(nstudyparameters-1,iterators,study_parameters)
 
        #Update set of hyperparameters
        temp = []
        for iparam in range(nstudyparameters):
            ival = iterators[iparam]
            val = study_parameters[iparam][ival]
            
            temp.append(val)

        config.append(temp)

    return config

###########################################################################################################

#                            RUNNING AN ESN

###########################################################################################################

def RunturbESN(esn, u_train: Union[np.ndarray, torch.Tensor] = None, 
                    y_train: Union[np.ndarray, torch.Tensor] = None, 
                    y_test: Union[np.ndarray, torch.Tensor] = None, 
                    pred_init_input: Union[np.ndarray, torch.Tensor] = None, 
                    u_test: Union[np.ndarray, torch.Tensor] = None, 
                    index_list_auto: list = [], index_list_teacher: list = []) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' Runs the turb Echo State Network (turbESN) with the specified parameters. 
        The training and validation/testing data must be specfied in the argument or in the esn object.
    
        INPUT: 
            esn             - ESN class object containing the reservoir parameters and training & validation/testing data sets.
            u_train         - training input 
            y_train         - training output
            u_test          - testing input
            y_test          - testing output
            pred_init_input - initial input in testing phase from which prediction start

        RETURN:
            mse_train - mean square error of (teacher forced!) reservoir output (to training target data set y_train) in the training phase. Mean w.r.t. timestep- & mode-axis. 
            mse_test  - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep- & mode-axis. 
            y_pred    - reseroir outputs, produced by the given reservoir specified in esn.
    '''

    torch.manual_seed(esn.randomSeed)  
    logging.basicConfig(format=_LOGGING_FORMAT, level= esn.logging_level)
    
    #---------------------------------------
    # Check whether user data is compatiable 
    # (data from esn object is assumed to be correct)
    #---------------------------------------
    assert None not in [u_train, y_train, y_test] or None not in [esn.u_train, esn.y_train, esn.y_test], "Error: u_train, y_train or y_test not specified."

    if None not in [u_train, y_train, y_test]:
        for data in [u_train, y_train]:
            if esn.trainingLength == data.shape[0]:
                logging.error('Training input/output time dimension ({0}) does not match ESN trainingLength ({1}).'.format(data.shape[0],esn.trainingLength))


    if y_test is not None:
        if esn.testingLength == y_test.shape[0]:
            logging.error('Testing Output time dimension ({0}) does not match ESN testingLength ({1}).'.format(y_test.shape[0],esn.testingLength))


    if esn.mode != _ESN_MODES[0]:
        assert u_test is not None or esn.u_test is not None, "Error: u_test not specified"
        if u_test is not None:
            if esn.testingLength == u_test.shape[0]:
                logging.error('Testing input time dimension ({0}) does not match ESN testingLength ({1}).'.format(u_test.shape[0],esn.testingLength))


    if pred_init_input is not None:
        if pred_init_input.dtype is not _DTYPE:
            pred_init_input = torch.as_tensor(pred_init_input, dtype = _DTYPE)
        if pred_init_input.device is not esn.device:
            pred_init_input.to(esn.device)

    if None not in [u_train, y_train, y_test]:
        esn.SetTrainingData(u_train=u_train, y_train=y_train)
        esn.SetTestingData(y_test=y_test, pred_init_input=pred_init_input, u_test=u_test)

    esn.toTorch()
    
    #---------------------------------------
    # ESN
    #---------------------------------------

    #1. Create Random Matrices
    esn.createWeightMatrices()

    #2. Training Phase 
    esn.x_fit = esn.propagate(u = esn.u_train, transientTime = esn.transientTime)
    esn.fit(X  = esn.x_fit, y = esn.y_train[esn.transientTime:])
    
    mse_train = ComputeMSE(y_test = esn.y_train[esn.transientTime:], y_pred =  (esn.Wout@esn.x_fit).T)

    if np.isnan(esn.Wout).any() or mse_train is None:
        logging.error("Reservoir {0}: while fitting the model, an error occured. Assuming default values.".format(esn.id))
        mse_train = _MSE_DEFAULT
        mse_test = _MSE_DEFAULT
        y_pred = torch.zeros([esn.testingLength, esn.n_output], device = esn.device, dtype = _DTYPE)

        return mse_train, mse_test, y_pred
  
    #3. Prediction Phase (Default:'auto')
    if esn.mode == _ESN_MODES[1]:
        y_pred, esn.x_pred = esn.teacherforce(X = esn.x_fit, testingLength = esn.testingLength)
  
    elif esn.mode == _ESN_MODES[2]:
        #TO DO: for now semi-teacher is only possible for n_input = n_output
        assert len(index_list_auto) + len(index_list_teacher) == esn.n_input,'index_list_auto and index_list_teacher do not add up to n_input.'
        y_pred, esn.x_pred = esn.semiteacherforce(X = esn.x_fit, testingLength = esn.testingLength, 
                                                  index_list_auto = index_list_auto, index_list_teacher = index_list_teacher, 
                                                  u_test = esn.u_test)
  
    else:
        y_pred, esn.x_pred = esn.predict(X = esn.x_fit, testingLength = esn.testingLength)

    mse_test = ComputeMSE(y_test = esn.y_test,y_pred = y_pred)
    
    return mse_train, mse_test, y_pred

###########################################################################################################

#                            ERROR METRICS

###########################################################################################################

def ComputeMSE(y_test: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''
    Computes the mean square error between target data y_test and prediction data y_pred.

    INPUT:
        y_pred - reseroir outputs
        y_test - validation output

    OUTPUT:
        mean square error of y_pred w.r.t. timestep-axis.
    '''
    
    logging.debug('Computing MSE')

    if y_test.dtype is not _DTYPE:
        y_test = torch.as_tensor(y_test, dtype = _DTYPE)
        
    if y_pred.dtype is not _DTYPE:
        y_pred = torch.as_tensor(y_pred, dtype = _DTYPE)
            
    return torch.mean((y_test-y_pred)**2, dim = 1)
#--------------------------------------------------------------------------
def ComputeR2(y_test: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''
    Computes the R2  between target data y_test and prediction data y_pred.
    R2 is defined as: 1 - squared error/(y_true - mean(y_true))

    INPUT:
        y_test - validation/testing/ true output
        y_pred - reseroir outputs

    OUTPUT:
        r2 w.r.t. both timestep- & mode-axis.
    '''
    
    if y_test.dtype is not _DTYPE:
        y_test = torch.as_tensor(y_test, dtype = _DTYPE)
        
    if y_pred.dtype is not _DTYPE:
        y_pred = torch.as_tensor(y_pred, dtype = _DTYPE)
            
    res = torch.sum((y_test - y_pred)**2,dim (0,1))
    mean = torch.mean(y_test, dim = 0)
    return 1 - res / torch.sum((y_pred - mean)**2, dim = (0,1))

#--------------------------------------------------------------------------
def ComputeWassersteinDistance(y_test: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''
    Computes the Wasserstein distance between target data y_test and prediction data y_pred.

    INPUT:
        y_test - validation/testing/ true output
        y_pred - reseroir outputs

    OUTPUT:
        Wasserstein distance w.r.t. both timestep- & mode-axis.
    '''

    hist_test, bins = torch.histogram(y_test, bins = 40)
    hist_pred,_ = torch.histogram(y_pred, bins = 40, range=(float(bins[0]),float(bins[-1])))

    return wasserstein_distance(hist_pred/torch.sum(hist_test), hist_test/torch.sum(hist_test))

###########################################################################################################

#                            SAVING ESN STUDY

###########################################################################################################

def CreateHDF5Groups(filepath: str, esn_ids: list, nstudy: int):
    '''Initializes the ESN study hdf5 file structure (shown below).  

    INPUT:
       filepath   - path to which the hdf5 file of the ESN study will be saved to
       esn_ids    - list of ESN IDs (usually these corresponds to different RNG seeds. For example esn_ids = range(nseeds) for nseeds no. RNG seeds.)
       nstudy     - number of different reservoir setting that are studied.
       
    FH added 15.03.2021
    '''
    #hdf5-Structure:
    #- ESN ID1
    #   - studyID1
    #       - study_dict
    #       - y_pred
    #       - mse
        
    #   - studyID2
    #       ...  
    #   - ...
    #- ESN ID2
    #   - studyID1
    #       ...
    #   - studyID2
    #   - ...
    #- ...
        
    with h5py.File(filepath, 'a') as f:
        for esn_id in esn_ids:
            G = f.create_group(str(esn_id))
            for studyID in range(nstudy):
                G.create_group(str(studyID))

#--------------------------------------------------------------------------
def SaveStudy(filepath: str, esn_id: int, 
              studyID: int, study_dict: dict, 
              y_pred: torch.Tensor, mse_train: torch.Tensor, 
              mse_test: torch.Tensor, f =None, wass_dist = None):
    '''Saves the ESN parameters from esn_params into a hdf5 file.
       The h5py file has to be init. with ReadModel (saving the fix parameters) before calling this function!
       
       INPUT:
          filepath   - path to which the hdf5 file of the ESN study is saved to
          esn_id     - ESN ID (usually corresponds to a RNG seed)
          studyID    - ID specifying the study
          study_dict - dictionary specifying the studied parameter configuration
          y_pred     - reseroir outputs
          mse_train  - mean square error of (teacher forced training) reservoir output (to training target dataset y_train) in the training phase. Mean w.r.t. timestep-axis. 
          mse_test   - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep-axis. 
          f          - opened hdf5.File where study will be saved to. If None, hdf5 file in filepath will be opened
          wass_dist  - wasserstein distance of testing data (optional)
       '''
           
    #HDF5-Structure:
        #- ESN ID1
        #   - studyID1
        #       - study_dict
        #       - y_pred
        #       - mse
        
        #   - studyID2
        #       ...  
        #   - ...
        #- ESN ID2
        #   - studyID1
        #       ...
        #   - studyID2
        #   - ...
        #- ...
        
    toClose = False
    logging.info('Saving study to {0}'.format(filepath))

    if f is None:
        assert os.path.isfile(filepath), "Error: The file {0} does not exist. Did you initialize the file with util.CreateHDF5Groups ?".format(filepath)
        f = h5py.File(filepath, 'a')
        toClose = True



    if '/' + str(esn_id)+'/' in f:
        G_esn_id = f.get(str(esn_id))
    else:
        G_esn_id = f.create_group(str(esn_id))

    if '/' + str(esn_id) +'/' + str(studyID) + '/' in f:
        G_study = G_esn_id.get(str(studyID))
    else:
        G_study = G_esn_id.create_group(str(studyID))
    

    for param in study_dict.keys():
        G_study.attrs[param] = study_dict[param]  
    
    #----------------------------------
    #  Datasets
    #----------------------------------    
    G_study.create_dataset('y_pred', data = y_pred, compression = 'gzip', compression_opts = 9)
    G_study.create_dataset('mse_train', data = mse_train, compression = 'gzip', compression_opts = 9)    
    G_study.create_dataset('mse_test', data = mse_test, compression = 'gzip', compression_opts = 9)

    if wass_dist is not None:
        G_study.create_dataset('wass_dist', data = wass_dist)

###########################################################################################################

#                            READING AN ESN STUDY

###########################################################################################################
#--------------------------------------------------------------------------
# FH added 28.02.2021
#--------------------------------------------------------------------------
def CreateStudyConfigArray(study_parameters: Union[list,tuple], study_dicts: dict) -> np.ndarray:
    ''' Computes an array, which gives the parameter configuration/setting for the corresponding study.
        
        INPUT: 
            study_parameters - list/tuple of strings specifying which parameters were studied. E.g. when reservoir size and density are studied  study_parameters = ['n_reservoir', 'reservoirDensity']
            study_dicts      - dictionary specifying the study parameter configuration

        RETURN:
            - config - array indicating the parameter setting for given study
        '''

    nparam = len(study_parameters)      #no. different parameters that are studied
    nstudy = len(study_dicts)           #no. studies/ parameter settings that were conducted
    config = np.empty([nstudy,nparam])

    for ii in range(nstudy):
        config_dict  =study_dicts[ii]
        for pp in range(nparam):
            key = study_parameters[pp]
            config[ii,pp] = config_dict[key]
            
    return config

#--------------------------------------------------------------------------
def ReadStudy(filepath: str, study_parameters: Union[list,tuple], nstudy: int = None, read_pred: bool = False, esn_ids: list = None, esn_id: int = None) -> Tuple[np.ndarray,np.ndarray,np.ndarray, np.ndarray]:
    '''Imports the results of the ESN study. 
        
        INPUT:
            filepath         - path to which the hdf5 file of the ESN study was saved to
            study_parameters - list/tuple of strings specifying which parameters were studied. E.g. when reservoir size and density are studied: study_parameters = ['n_reservoir', 'reservoirDensity']
            nstudy           - number of different reservoir setting that were studied. If nstudy = None, the number is deduced from the file.
            read_pred         - boolean specifying whether to import ESN predictions y
            esn_ids          - list of ESN IDs (usually these corresponds to different RNG seeds. For example esn_ids = range(nseeds) for nseeds no. RNG seeds.)
            esn_id           - ESN ID (usually corresponds to a RNG seed)
            
        RETURN:
            mse_train   - mean square training error of reservoir output (to validation data set y_train). Mean w.r.t. timestep-axis. 
            mse_test    - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep-axis. 
            y_pred      - reseroir outputs, for each study parameter setting of the study 
            config      - array indicating the parameter setting for given study
    '''

    assert esn_id is not None or esn_ids is not None, "esn_id and esn_ids cant both be None."
    assert os.path.isfile(filepath),"Error: File {0} not found.".format(filepath)

    if read_pred:
        logging.debug("Reading reservoir outputs.")

    #read specified seed
    if esn_id is not None:  
        with h5py.File(filepath,'r') as f:

            mse_train, mse_test, y_pred, study_dicts = [], [], [], []
            G_esn_id = f.get(str(esn_id))
            
            #if user does not specify study number, the number is inferred from no. subgroups
            if nstudy is None:
                nstudy = len(G_esn_id.keys())
            
            for istudy in range(nstudy):
                study_dict = {}
                G_study = G_esn_id.get(str(istudy))

                if mse_is_array:                                                         #20.10.21: if mse was saved as array, this can now be read
                    mse_train.append(np.array(G_study.get('mse_train')))
                    mse_train.append(np.array(G_study.get('mse_test')))
                else:
                    mse_train.append(G_study.attrs['mse_train'])
                    mse_test.append(G_study.attrs['mse_test'])

                if read_pred:
                    y_pred.append(np.array(G_study.get('y_pred')))

                for name in study_parameters:
                    study_dict[name] = G_study.attrs[name]
                    
                study_dicts.append(study_dict)
            
        config = CreateStudyConfigArray(study_parameters, study_dicts)
        return np.array(mse_train), np.array(mse_test), np.array(y_pred), config

    #read all specified seeds
    else:
        with h5py.File(filepath,'r') as f:
            
            mse_train, mse_test, y_pred, study_dicts = [], [], [], []

            ii = 0
            for esn_id in esn_ids:
                G_esn_id = f.get(str(esn_id))
            
                #if user does not specify study number, the number is inferred from no. subgroups
                if nstudy is None:
                    nstudy = len(G_esn_id.keys())
            
                MSE_train, MSE_test, Y_pred = [], [], []
                for istudy in range(nstudy):
                    G_study = G_esn_id.get(str(istudy))

                    if ii == 0:
                        study_dict = {}
                        for name in study_parameters:
                            study_dict[name] = G_study.attrs[name]
                        study_dicts.append(study_dict)
            
                    if mse_is_array:
                        MSE_train.append(np.array(G_study.get('mse_train')))
                        MSE_test.append(np.array(G_study.get('mse_test')))
                    else: 
                        MSE_train.append(G_study.attrs['mse_train'])
                        MSE_test.append(G_study.attrs['mse_test'])

                    if readPred:
                        Y_pred.append(np.array(G_study.get('y_pred')))

                mse_train.append(MSE_train)
                mse_test.append(MSE_test)
                y_pred.append(Y_pred)
                
                ii+=1
                    
        config = CreateStudyConfigArray(study_parameters, study_dicts)
        return np.array(mse_train), np.array(mse_test), np.array(y_pred), config


###########################################################################################################

#                            EXPERIMENTAL

###########################################################################################################

def InitRandomSearchStudyOrder(nstudy: int, study_tuple: tuple, HP_range_dict: dict = {}) -> list:
    ''' Initializes the HP study parameter settings for a random search. 
        Each of the nstudy settings is given by an entry of the returned list config.

    INPUT:
        nstudy           - number of different reservoir setting that should be studied. 
        study_tuple      - tuple specifying the parameters that are studied
        HP_range_dict    - dictionary containing the interval of each hyperparameter (HP must be specified in study_set) 

    RETURN: 
        config - array indicating the parameter setting for given study
    '''


    if not HP_range_dict:
        HP_range_dict = ESN.hyperparameter_intervals()

    config = []

    for ii in range(nstudy):
        setting = []
        iparam = 0
        for param in study_tuple:
            if param == 'regressionParameters':
                setting.append([loguniform.rvs(HP_range_dict[param][0],HP_range_dict[param][1],size = 1)[0]])
            elif param == 'n_reservoir':
                setting.append(int(loguniform.rvs(HP_range_dict[param][0],HP_range_dict[param][1],size = 1)[0]))
            else:
                setting.append(loguniform.rvs(HP_range_dict[param][0],HP_range_dict[param][1],size = 1)[0])
            
            iparam +=1
        config.append(setting)
    
    return config

#--------------------------------------------------------------------------
