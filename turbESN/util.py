#turbESN
from .core import ESN
from .cross_validation import CrossValidation
from ._modes import (_DTYPE, _DEVICE, _ESN_MODES, _WEIGTH_GENERATION, _EXTENDED_STATE_STYLES, _LOSS_DEFAULT)


#backends
import numpy as np
import torch

#save data structure
import h5py 
import json

#misc
import sys
import os
from copy import deepcopy
import logging
import logging.config

from typing import Union, Tuple, List

from scipy.stats import wasserstein_distance
from scipy.stats import loguniform, uniform


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


###########################################################################################################

#                             PRE-PROCESSING/ IMPORTING

###########################################################################################################
#--------------------------------------------------------------------------
def prepare_auto_data(data: Union[np.ndarray, torch.Tensor],
                         n_input: int, 
                         trainingLength: int, 
                         testingLength: int, 
                         esn_start: int, 
                         esn_end: int,
                         validationLength: int=None) -> Union[Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
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
    
    data = torch.as_tensor(data, dtype=_DTYPE)

    data_esn = data[esn_start-1:esn_end,:n_input]
    u_train = data_esn[0:trainingLength,:]
    y_train = data_esn[1:trainingLength+1,:]
    
    u_test = data_esn[trainingLength:trainingLength+testingLength,:]
    y_test = data_esn[trainingLength+1:trainingLength+testingLength+1,:]

    if validationLength is not None:
        u_val = data_esn[trainingLength+testingLength:trainingLength+testingLength+validationLength,:]
        y_val = data_esn[trainingLength+testingLength+1:,:]
    else:
        u_val = None
        y_val = None

    return u_train.to(_DTYPE), y_train.to(_DTYPE), u_test.to(_DTYPE), y_test.to(_DTYPE), u_val.to(_DTYPE), y_val.to(_DTYPE)
 
#--------------------------------------------------------------------------
def prepare_teacher_data(data_in: Union[np.ndarray, torch.Tensor], 
                        data_out: int, 
                        n_input: int, 
                        n_output: int, 
                        trainingLength: int, 
                        testingLength: int, 
                        esn_start: int, 
                        esn_end: int,
                        validationLength: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' Prepares the input and output training and testing/validation data set for the teacher forced ESN case, i.e. where the
        reservoir input is a teacher signal. 

    INPUT:
        data_in        - data array to be used for the ESN. The shape must be (data_timesteps, n_input)
        data_out       - data array to be used for the ESN. The shape must be (data_timesteps, n_output)
        n_input        - input dimensions/ no. input modes
        n_output       - output dimensions/ no. output modes
        trainingLength - no. time steps for the training data set
        testingLength  - no. time steps for the testing/validation data set
        validationLength -  no. time steps for the validation data set
        esn_start      - Index of the original data, at which the training output y_train will begin. 
        esn_end        - Index of the original data, at which the testing/validation output y_test will end.

    RETURN:
        u_train - training input data set. Shape: (trainingLength, n_input)
        y_train - training output data set. Shape: (trainingLength, n_output)
        u_test  - testing/validation input data set.  Shape: (testingLength, n_input)
        y_test  - testing/validation output data set. Shape: (testingLength, n_output)
        u_val   - validation input data set (not used in auotnomous predictor mode). Shape: (validationLength, n_input)
        y_val   - validation output data set. Shape: (validationLength, n_output)
    '''

    data_in = torch.as_tensor(data_in, dtype=_DTYPE)
    data_out = torch.as_tensor(data_out, dtype=_DTYPE)
    
    data_in_esn = data_in[esn_start-1:esn_end,:n_input]
    data_out_esn = data_out[esn_start-1:esn_end,:n_output]

    u_train = data_in_esn[0:trainingLength]
    y_train = data_out_esn[0:trainingLength]
    
    u_test = data_in_esn[trainingLength:trainingLength+testingLength]
    y_test = data_out_esn[trainingLength:trainingLength+testingLength]

    if validationLength is not None:
        u_val = data_in_esn[trainingLength+testingLength:trainingLength+testingLength+validationLength]
        y_val = data_out_esn[trainingLength+testingLength:trainingLength+testingLength+validationLength]

    else:
        u_val = None
        y_val = None

    return u_train.to(_DTYPE), y_train.to(_DTYPE), u_test.to(_DTYPE), y_test.to(_DTYPE), u_val.to(_DTYPE), y_val.to(_DTYPE)
    #return torch.as_tensor(u_train, dtype = _DTYPE), torch.as_tensor(y_train, dtype = _DTYPE),torch.as_tensor(u_test, dtype = _DTYPE),torch.as_tensor(y_test, dtype = _DTYPE), torch.as_tensor(u_val, dtype = _DTYPE),torch.as_tensor(y_val, dtype = _DTYPE)


#--------------------------------------------------------------------------
def init_study_order(nsetting: int, study_parameters: list) -> list:
    ''' Initializes the study parameter settings into an array.

    INPUT:
        nsetting         - number of different reservoir settings that were studied. If nsetting = None, the number is deduced from the file.
        study_parameters - list specifying the values of the parameters that are studied

    RETURN: 
        config - list indicating the parameter setting for given study
    ''' 
    
    assert len(study_parameters) != 0,'study_parameters are empty. Did you forget to specify the range of the studied HP?'
    assert nsetting > 0, 'nsetting ({0}) must be > 0'.format(nsetting)

    def recursion(iparam: int, iterators: np.ndarray, study_tuple: tuple):
        ''' Iterates the iterators which are used to change the hyperparmeters. 
            Makes sure that all combinations of the parameters in study_tuple are used.

            INPUT:
                iparam           - index which defines the hyperparameter that is changed
                iterators        - iterator that defines the no. ESNs that have been run with changing one hyperparameter (study_tuple[iparam]) 
                study_tuple      - tuple specifying the range of the parameters that are studies

        FH 22.03.21: moved function from esn_user_mp.py/esn_user_thread.py to this file
        '''
        
        #if iterator associated with iparam hasn't reached final value --> increment
        if iterators[iparam] < len(study_tuple[iparam])-1:
            iterators[iparam] +=1
                
        #if iterator associated with iparam has reached final value --> reset iterator and increment/reset higher level iterator associated with iparam-1.
        else:
            iterators[iparam] = 0
            recursion(iparam-1,iterators,study_tuple)

    config = []
    nstudyparameters = len(study_parameters)
    iterators = np.zeros([nstudyparameters], dtype=int)

    for itotal in range(nsetting):
        #Update iterators:      
        if itotal == 0:
            pass
        else:
            recursion(nstudyparameters-1,iterators,study_parameters)
 
        #Update set of hyperparameters
        temp = []
        for iparam in range(nstudyparameters):
            ival = iterators[iparam]
            val = study_parameters[iparam][ival]
            
            temp.append(val)

        config.append(temp)

    return config

#--------------------------------------------------------------------------
def init_random_search(nsetting: int, study_tuple: tuple, limits: list = []) -> list:
    ''' Initializes the hyperparameter study parameter settings for a random search. 
        Each of the nsetting settings is given by an entry of the returned list config.

    INPUT:
        nsetting         - number of different reservoir setting that should be studied
        study_tuple      - tuple specifying the parameters that are studied
        limits           - user specified range values of hyperparameters in study_tuple: [val_min, val_max, nval, use_log]  
    RETURN: 
        config   - array indicating the parameter setting for given study
        nsetting - number of different reservoir setting that should be studied
    '''

    if len(study_tuple) == len(limits):               
        HP_range_dict = dict(zip(study_tuple,limits))

    else:
        # default: use standard values & log spacing 
        HP_range_dict = {param: list(HP_dict[param]["RANGE"])+[0,True] for param in study_tuple}

    config = []

    for ii in range(nsetting):
        setting = []
        for iparam,param in enumerate(study_tuple):

            min_val, max_val, _, use_log_scale = HP_range_dict[param]

            if use_log_scale:
                value = loguniform.rvs(min_val,max_val,size = 1)[0]
            else:
                 value = uniform.rvs(min_val,max_val,size = 1)[0]

            if param == 'n_reservoir':
                value = int(value)

            setting.append(value)
        config.append(setting)
    
    return config, nsetting

#--------------------------------------------------------------------------
def init_grid_search(study_tuple, limits, lists):
    ''' Initializes the hyperparameter study parameter settings for a grid search. 

    INPUT:
        study_tuple      - tuple specifying the parameters that are studied
        limits           - user specified range values of hyperparameters in study_tuple: [val_min, val_max, nval, use_log]  
        limits           - user specified values of the hyperparameters in study_tuple:   [val1, val2, val3,...] 
        
    RETURN: 
        config   - array indicating the parameter setting for given study
        nsetting - number of different reservoir setting that should be studied
    '''

    study_parameters = []
    for ii,limit in enumerate(limits): 
        
        # HP grid values are read from user specified values
        if len(lists[ii]) is not None and len(lists[ii]) != 0:
            param_val = lists[ii]

        # HP grid is computed from user specified ranges
        else:
            assert len(limit) == 4, "Error: limits must be of style [value_min, value_max, nvals, use_log]!"
            xmin, xmax, nx, use_log = limit

            if use_log:
                param_val = np.logspace(xmin,xmax,nx)
            else:
                param_val = np.linspace(xmin,xmax,nx)

        study_parameters.append(param_val)   

    assert len(study_parameters) == len(study_tuple),f"Error: Length of study_tuple ({len(study_tuple)})  does not match no. study parameters ({len(study_parameters)})!\n"
    nsetting = np.prod([len(param_arr) for param_arr in study_parameters])
    config   = init_study_order(nsetting, tuple(study_parameters)) 

    return config, nsetting

#--------------------------------------------------------------------------
#FH 25.05.2022: added minmax_scaling
def minmax_scaling(x, x_min=None,x_max=None, dataScaling=1):
    '''
    Applies min-max scaling to data x & shifts data to [-dataScaling,dataScaling]. 
    If x_min,x_max not given, they are compute based on min/max values along time axis (axis 0)

    INPUT:
        x           - data, shape: (timesteps, modes) 
        x_min       - min value of data x
        x_max       - max value of data x 
        dataScaling - scaling factor

    RETURN:
        x_scaled - x scaled to interval [-dataScaling, dataScaling]
    '''
    arr_type = type(x)

    if x_min is None:
        if arr_type == np.ndarray:
            x_min = x.min(axis=0)
        elif arr_type == torch.Tensor:
            x_min = torch.amin(x,dim=0)

    if x_max is None:
        if arr_type == np.ndarray:
            x_max = x.max(axis=0)
        elif arr_type == torch.Tensor:
            x_max = torch.amax(x,dim=0)
            

    x_scaled = (x - x_min)/(x_max- x_min)
    x_scaled -= 0.5
    x_scaled *=2*dataScaling

    return x_scaled
#--------------------------------------------------------------------------

def undo_minmax_scaling(x, x_min,x_max, dataScaling=1):
    '''
    Reverses min-max scaling.

    INPUT:
        x           - minmax-scaled data, shape: (timesteps, modes) 
        x_min       - min value of data unscaled x
        x_max       - max value of data unscaled x 
        dataScaling - scaling factor

    RETURN:
        x_unscaled - data in original range(x_min,x_max)
    '''

    return 0.5*(x/dataScaling+1)*(x_max-x_min)+x_min

#--------------------------------------------------------------------------
#FH 30.03.2022: added check_user_input (prev. in run_turbESN)
def check_user_input(esn, u_train: Union[np.ndarray, torch.Tensor]=None, 
                    y_train:Union[np.ndarray, torch.Tensor]=None, 
                    y_test:Union[np.ndarray, torch.Tensor]=None, 
                    test_init_input:Union[np.ndarray, torch.Tensor]=None, 
                    u_test:Union[np.ndarray, torch.Tensor]=None, 
                    u_val:Union[np.ndarray, torch.Tensor]=None, 
                    y_val:Union[np.ndarray, torch.Tensor]=None,
                    val_init_input: Union[np.ndarray, torch.Tensor]=None, 
                    u_pre_val:Union[np.ndarray, torch.Tensor]=None,
                    loss_func:dict=None):

    '''Checks data provided by user. The ESN object might get modified by this method.'''

    #---------------------------------------
    # Check whether user data is compatiable 
    # (data from esn object is assumed to be correct)
    #---------------------------------------

    assert None not in [u_train, y_train, y_test] or None not in [esn.u_train, esn.y_train, esn.y_test], "Error: u_train, y_train or y_test not specified."

    if None not in [u_train, y_train, y_test]:
        for data in [u_train, y_train]:
            if esn.trainingLength != data.shape[0]:
                logger.error('Training input/output time dimension ({0}) does not match ESN trainingLength ({1}).'.format(data.shape[0],esn.trainingLength))


    if y_test is not None:
        if esn.testingLength != y_test.shape[0]:
            logger.error('Testing Output time dimension ({0}) does not match ESN testingLength ({1}).'.format(y_test.shape[0],esn.testingLength))


    if esn.mode != _ESN_MODES[0]:
        assert u_test is not None or esn.u_test is not None, "Error: u_test not specified"
        if u_test is not None:
            if esn.testingLength != u_test.shape[0]:
                logger.error('Testing input time dimension ({0}) does not match ESN testingLength ({1}).'.format(u_test.shape[0],esn.testingLength))


    if test_init_input is not None:
        if test_init_input.dtype is not _DTYPE:
            test_init_input = torch.as_tensor(test_init_input, dtype = _DTYPE)
        if test_init_input.device is not esn.device:
            test_init_input.to(esn.device)

    if None not in [u_train, y_train, y_test]:
        esn.SetTrainingData(u_train=u_train, y_train=y_train)
        esn.SetTestingData(y_test=y_test, test_init_input=test_init_input, u_test=u_test)

    if None not in [u_val, y_val]:
        esn.SetValidationData(y_val=y_val, u_val=u_val, val_init_input=val_init_input,u_pre_val=u_pre_val)

    if loss_func is not None:
        esn.loss_func = loss_func
    else:
        if esn.loss_func is None:
            esn.loss_func = dict(mse=compute_mse)

    esn.to_torch()
###########################################################################################################

#                            RUNNING AN ESN

###########################################################################################################

def run_turbESN(esn, 
                u_train: Union[np.ndarray, torch.Tensor]=None, 
                y_train: Union[np.ndarray, torch.Tensor]=None, 
                y_test: Union[np.ndarray, torch.Tensor]=None, 
                test_init_input: Union[np.ndarray, torch.Tensor]=None, 
                u_test: Union[np.ndarray, torch.Tensor]=None, 
                u_val: Union[np.ndarray, torch.Tensor]=None,
                y_val: Union[np.ndarray, torch.Tensor]=None,
                val_init_input:Union[np.ndarray, torch.Tensor]=None, 
                u_pre_val:Union[np.ndarray,torch.Tensor]=None,
                index_list_auto: list = [], index_list_teacher: list = [],
                recompute_Win:bool=True,
                recompute_Wres:bool=True,
                recompute_Wfb:bool=True,
                loss_func:dict=None) -> Tuple[dict,torch.Tensor,torch.Tensor]:

    ''' Runs the turbulent Echo State Network (turbESN) with the specified parameters. 
        The training and testing data must be specfied in the argument or in the esn object.
        Optional: validation data (third dataset for cross validation, note naming convention (commonly: training/validation/testing, here: training/testing/validation))
    
        INPUT: 
            esn                - ESN class object. Contains the reservoir parameters. May contain training, testing and validation data sets.
            u_train            - training input (optional if esn.u_train not None)
            y_train            - training output (optional if esn.y_train not None)
            u_test             - testing input (optional if esn.u_test not None)
            y_test             - testing output (optional if esn.y_test not None)
            test_init_input    - initial input in testing phase from which prediction start
            u_val              - validation input (optional)
            y_val              - validation output (optional)
            val_init_input     - initial input in validation phase from which prediction starts (only if esn.mode = _ESN_MODES[1])
            u_pre_val          - data used to build up reservoir state for start of validation phase (if None, last instance of self.x_test will be used)
            index_list_auto    - indices of the modes which are passed back as new input (only if esn.mode = _ESN_MODES[2])
            index_list_teacher - indices of the modes which are supplied by a teacher signal (only if esn.mode = _ESN_MODES[2])
            recompute_Win      - whether input matrix must be initialized
            recompute_Wres     - whether reservoir matrix must be initialized
            recompute_Wfb      - whether feedback matrix must be initialized
            loss_func          - dict containing the loss function used for prediction evalutation
            
        RETURN:
            loss_dic    - dictionary containing the losses specified in loss_func
            y_pred_test - reseroir outputs of testing phase
            y_pred_val  - reseroir outputs of validation phase
            
    '''
    torch.manual_seed(esn.randomSeed)  

    check_user_input(esn=esn,
                    u_train=u_train,
                    y_train=y_train,
                    u_test=u_test,
                    y_test=y_test,
                    test_init_input=test_init_input,
                    u_val=u_val,
                    y_val=y_val,
                    val_init_input=val_init_input,
                    u_pre_val=u_pre_val,
                    loss_func=loss_func)

    loss_dict = {}
    ##############################################
    # Run ESN
    ##############################################
    
    #----------------------------------------------------------
    #1. Create Random Matrices
    #----------------------------------------------------------
    if recompute_Win:
        esn.createInputMatrix()
    if recompute_Wres:
        esn.createReservoirMatrix()
    if esn.use_feedback and recompute_Wfb:
        esn.createFeebackMatrix()

    #----------------------------------------------------------
    #2. Training Phase
    #----------------------------------------------------------
    logger.debug('Training ESN')
    esn.x_train=esn.propagate(u = esn.u_train, transientTime = esn.transientTime,y=esn.y_train)
    esn.fit(X=esn.x_train, y=esn.y_train[esn.transientTime:])
    loss_dict['mse_train']  = compute_mse(y_true=esn.y_train[esn.transientTime:], y_pred=(esn.Wout@esn.x_train).T)
    
    # Check for errors in training
    #-----------------------------
    if np.isnan(esn.Wout).any() or loss_dict['mse_train']  is None:
        logger.error("Reservoir {0}: while fitting the model, an error occured. Assuming default values.".format(esn.id))
        loss_dict['mse_train']  = torch.tensor([_LOSS_DEFAULT for _ in range(int(esn.trainingLength-esn.transientTime))],device = esn.device,dtype = _DTYPE)
        default_test = torch.tensor([_LOSS_DEFAULT for _ in range(esn.testingLength)],device = esn.device,dtype = _DTYPE)       
        default_val = torch.tensor([_LOSS_DEFAULT for _ in range(esn.validationLength)],device = esn.device,dtype = _DTYPE)
    
        loss_test, loss_val = [], []
        if esn.loss_func is not None:
            for loss_label in esn.loss_func.keys():
                loss_dict[loss_label+'_test'] = default_test
                
                if None not in [esn.u_val, esn.y_val]:
                    loss_dict[loss_label+'_val']  = default_val
                
        y_pred_test = torch.zeros([esn.testingLength, esn.n_output], device = esn.device, dtype = _DTYPE)
        y_pred_val = torch.zeros([esn.validationLength, esn.n_output], device = esn.device, dtype = _DTYPE)

        return loss_dict, y_pred_test, y_pred_val

    #----------------------------------------------------------
    #3. Prediction/Testing Phase (Default:'auto')
    #----------------------------------------------------------
    logger.debug('Testing ESN')
    if esn.mode == _ESN_MODES[0]:
        y_pred_test, esn.x_test = esn.predict(X = esn.x_train, testingLength = esn.testingLength)

    elif esn.mode == _ESN_MODES[1]:
        y_pred_test, esn.x_test = esn.teacherforce(X = esn.x_train, testingLength = esn.testingLength)
  
    elif esn.mode == _ESN_MODES[2]:
        #TO DO: for now semi-teacher is only possible for n_input = n_output
        assert len(index_list_auto) + len(index_list_teacher) == esn.n_input,'index_list_auto and index_list_teacher do not add up to n_input.'
        y_pred_test, esn.x_test = esn.semiteacherforce(X = esn.x_train, testingLength = esn.testingLength, 
                                                  index_list_auto = index_list_auto, index_list_teacher = index_list_teacher, 
                                                  u_test = esn.u_test)
    else:
        raise NotImplementedError('Error: unkown mode {0}. Choices {1}'.format(mode, _ESN_MODES))                                
 
    #-------------------------------------------------------------------------
    #4. (optional) Validation Phase (for now only in auto & teacherforce mode)
    #-------------------------------------------------------------------------
    logger.debug('Validating ESN')
    y_pred_val = None
    
    if None not in [esn.u_val, esn.y_val]:
        
        validationLength = esn.y_val.shape[0]

        # initial state is build up from GT data (similar to testing phase)
        if esn.u_pre_val is not None:
            X_pre_val = esn.propagate(u = esn.u_pre_val, transientTime = esn.u_pre_val.shape[0]-1,y=esn.y_pre_val)

        # initial state taken from testing phase (no state initialization by GT data)
        else:
            X_pre_val = esn.x_test

            if esn.val_init_input is None:
                if esn.mode == _ESN_MODES[0]:
                    esn.val_init_input = esn.y_test[-1,:].reshape(1,esn.n_input)

        if esn.mode == _ESN_MODES[0]:
            y_pred_val, esn.x_val = esn.predict(X=X_pre_val,testingLength=validationLength,init_input=esn.val_init_input)

        elif esn.mode == _ESN_MODES[1]:
            y_pred_val, esn.x_val = esn.teacherforce(X=X_pre_val, testingLength = esn.validationLength, u=esn.u_val)
        else:
            raise NotImplementedError('Error: mode {0} not supported for validation. Choices {1}'.format(mode, _ESN_MODES[:2]))

    #-------------------------------------------------------------------------
    #5. Compute test & validation losses 
    #-------------------------------------------------------------------------
    if esn.loss_func is not None:
        for loss_label,loss_func in esn.loss_func.items():
            loss_dict[loss_label+'_test'] = loss_func(y_true=esn.y_test,y_pred=y_pred_test)

            if None not in [esn.u_val, esn.y_val]:
                loss_dict[loss_label+'_val'] = loss_func(y_true=esn.y_val,y_pred=y_pred_val)

    return loss_dict, y_pred_test, y_pred_val

#--------------------------------------------------------------------------
#FH 30.03.2022: added forward_validate_auto_ESN 
def forward_validate_auto_ESN(esn: ESN, cv: CrossValidation):
    ''' Runs ESN with k-fold forward walk validation scheme described in Lukosevicius et al. (2021)

    
    
    INPUT:
        esn    - ESN class object. Contains the reservoir parameters. May contain training, testing and validation data sets.
        cv     - CrossValidation class object  containing cross validation parameters and data
        
    RETURN:
        loss_dict      - dictionary containing the losses specified in esn.loss_func
        y_pred_test_cv - reseroir outputs of the testing phase for each fold
        y_pred_val_cv  - reseroir outputs of the validation phase for each fold
        '''

    max_folds = int(cv.n_folds - cv.n_training_folds - cv.n_validation_folds)
    transientTime0 = esn.transientTime

    loss_dict = dict()
    if esn.loss_func is not None:
        loss_dict['mse_train'] = list()
        for loss_label,loss_func in esn.loss_func.items():
            loss_dict[loss_label+'_test'] = list()
    
    y_pred_test_cv = torch.empty((max_folds,esn.testingLength,esn.n_output), dtype=_DTYPE)
    y_pred_val_cv  = torch.empty((max_folds,esn.validationLength,esn.n_output), dtype=_DTYPE)

    logger.debug(f'Starting forward validation w. {max_folds} folds')
    for ifold in range(max_folds):

        #----------------------------
        # indices of folds
        #----------------------------
        itrain_end = cv.n_training_folds+ifold
        itest_end = itrain_end + cv.n_testing_folds
        
        #----------------------------
        # length of folds
        #----------------------------
        sliding_trainingLength = int(cv.n_training_folds*cv.fold_length) + ifold*cv.fold_length
        sliding_transientTime = transientTime0 + ifold*cv.fold_length
        
        esn.SetTransientTime(sliding_transientTime)
        esn.trainingLength = sliding_trainingLength
        
        #----------------------------
        # create input/output data
        #----------------------------
        u_train = cv.data_folded_u[0:itrain_end].reshape(-1,esn.n_input)
        y_train = cv.data_folded_y[0:itrain_end].reshape(-1,esn.n_input)
        
        u_test = cv.data_folded_u[itrain_end:itest_end].reshape(-1,esn.n_input)
        y_test = cv.data_folded_y[itrain_end:itest_end].reshape(-1,esn.n_input)

        u_val = cv.data_folded_u[-1].reshape(-1,esn.n_input)
        y_val = cv.data_folded_y[-1].reshape(-1,esn.n_input)

        if ifold != max_folds -1:
            u_pre_val = cv.data_folded_u[itest_end:-1].reshape(-1,esn.n_input)
        else:
            u_pre_val = None  # validation data is directly preceeded by testing dataset

        #to do: implement more efficient k fold forward walk (precompute esn.x_train, efficient matrix inverse)

        #----------------------------
        #Run ESN
        #----------------------------
        loss_dict_single, y_pred_test, y_pred_val = run_turbESN(esn = esn, 
                                                                u_train = u_train, 
                                                                y_train = y_train, 
                                                                y_test = y_test, 
                                                                u_test = u_test, 
                                                                u_val=u_val, 
                                                                y_val=y_val,
                                                                u_pre_val=u_pre_val)

        if esn.loss_func is not None:
            loss_dict['mse_train'].append(loss_dict_single['mse_train'])
            for label in esn.loss_func.keys():
                loss_dict[label+'_test'].append(loss_dict_single[label+'_test'])
                loss_dict[label+'_val'].append(loss_dict_single[label+'_val'])

        y_pred_test_cv[ifold] = y_pred_test
        y_pred_val_cv[ifold]  = y_pred_val
        
    return loss_dict, y_pred_test_cv, y_pred_val_cv

###########################################################################################################

#                            ERROR METRICS

###########################################################################################################

def compute_mse(y_true: torch.Tensor, 
                y_pred: torch.Tensor, 
                axis: Union[int,tuple] = 1) -> torch.Tensor:
    '''
    Computes the mean square error between target data y_true and prediction data y_pred.

    INPUT:
        y_pred - reseroir outputs
        y_true - validation output
        axis   - axis over which MSE should be computed. 0: time-axis, 1: mode-axis, (0,1): both

    OUTPUT:
        mean square error of y_pred w.r.t. specified axis
    '''
    
    logger.debug('Computing MSE')

    if y_true.dtype is not _DTYPE:
        y_true = y_true.to(_DTYPE)
        
    if y_pred.dtype is not _DTYPE:
        y_pred = y_pred.to(_DTYPE)
            
    return torch.mean((y_true-y_pred)**2, dim = axis)

#--------------------------------------------------------------------------
def compute_nrmse(y_true: torch.Tensor, 
                  y_pred: torch.Tensor, 
                  axis: Union[int,tuple] = 1) -> torch.Tensor:
    '''
    Computes the normalized root mean square error between target data y_true and prediction data y_pred.

    INPUT:
        y_pred - reseroir outputs
        y_true - validation output
        axis   - axis over which MSE should be computed. 0: time-axis, 1: mode-axis, (0,1): both

    OUTPUT:
        mean square error of y_pred w.r.t. specified axis
    '''
    
    logger.debug('Computing NRMSE')

    if y_true.dtype is not _DTYPE:
        y_true = y_true.to(_DTYPE)
        
    if y_pred.dtype is not _DTYPE:
        y_pred = y_pred.to(_DTYPE)
            
    return torch.sqrt(torch.mean((y_true-y_pred)**2, dim = axis)/ (torch.std(y_pred) + 1e-6))


#--------------------------------------------------------------------------
def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''
    Computes the R2  between target data y_true and prediction data y_pred.
    R2 is defined as: 1 - (y_true -y_pred)**"/(y_true - mean(y_true))

    INPUT:
        y_true - validation/testing/ true output
        y_pred - reseroir outputs

    OUTPUT:
        r2 w.r.t. both timestep- & mode-axis.
    '''
    
    logger.debug('Computing R2')

    if y_true.dtype is not _DTYPE:
        y_true = torch.as_tensor(y_true, dtype = _DTYPE)
        
    if y_pred.dtype is not _DTYPE:
        y_pred = torch.as_tensor(y_pred, dtype = _DTYPE)
            
    res = torch.sum((y_true - y_pred)**2,dim (0,1))
    mean = torch.mean(y_true, dim = 0)
    return 1 - res / torch.sum((y_pred - mean)**2, dim = (0,1))

#--------------------------------------------------------------------------
def compute_wasserstein_distance(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''
    Computes the Wasserstein distance between target data y_true and prediction data y_pred.

    INPUT:
        y_true - validation/testing/ true output
        y_pred - reseroir outputs

    OUTPUT:
        Wasserstein distance w.r.t. both timestep- & mode-axis.
    '''


    logger.debug('Computing Wasserstein Distance')

    hist_test, bins = torch.histogram(y_true, bins = 40)
    hist_pred,_ = torch.histogram(y_pred, bins = 40, range=(float(bins[0]),float(bins[-1])))

    return wasserstein_distance(hist_pred/torch.sum(hist_test), hist_test/torch.sum(hist_test))

#--------------------------------------------------------------------------
# FH 28.08.2022: added normalized prediction error according to Tanaka et al. Phys. Rev. Res. 4 (2022)
def compute_normalized_prediction_error(y_true:torch.Tensor, 
                                        y_pred:torch.Tensor, 
                                        modal_mean:bool=True) -> torch.Tensor:
    """
    Computes the normalized prediction error (or its mean):
                  ||y_true-y_pred|| / <y_true**2>_t^(1/2)
    INPUT: 
        y_true     - validation/ testing/ true output
        y_pred     - reseroir outputs
        modal_mean - return modal mean of normalized prediction errror
    RETURN:
        normalized_prediction_error - modal mean of normalized prediction error
    """
    

    logger.debug('Computing NPE')

    testingLength,n_output = y_true.shape
    
    norm = (torch.mean(y_true**2+1e-6,dim=(0,)).reshape(1,n_output))**(1/2)
    normalized_prediction_error = torch.abs(y_true - y_pred)/norm
    
    if modal_mean:
        normalized_prediction_error = normalized_prediction_error.mean(dim=1)
    
    return normalized_prediction_error
    
###########################################################################################################

#                            SAVING ESN STUDY

###########################################################################################################

def create_hdf5_groups(filepath: str, seeds: Union[int,list,range], nsetting: int):
    '''Initializes the ESN study hdf5 file structure (shown below).  

    INPUT:
       filepath   - path to which the hdf5 file of the ESN study will be saved to
       nseed      - no. RNG seeds used in grid search
       nsetting   - no. ESN settings used in grid search
       
    FH added 15.03.2021
    '''
    #hdf5-Structure:
    #- seed1
    #   - setting1
    #       - study_dict
    #       - y_pred
    #       - mse
        
    #   - setting2
    #       ...  
    #   - ...
    #- seed2
    #   - setting1
    #       ...
    #   - setting2
    #   - ...
    #- ...
        
    if isinstance(seeds,int):
        seeds = range(seeds)
    
    with h5py.File(filepath, 'a') as f:
        
        if 'Study' in f:
            G_study = f['Study']
        else:
            G_study = f.create_group('Study')

        for seed in seeds:
            if f'{seed}' not in G_study:
                G_seed = G_study.create_group(f'{seed}')
            else:
                G_seed = G_study[f'{seed}']

            for isetting in range(nsetting):
                if f'{isetting}' not in G_seed:
                    G_seed.create_group(f'{isetting}')

#--------------------------------------------------------------------------
def save_study(filepath: str, 
              randomSeed: int, 
              isetting: int, 
              study_dict: dict, 
              y_pred_test: torch.Tensor,
              y_pred_val: torch.Tensor,
              loss_dict: dict,
              f =None):
    '''Saves the ESN parameters from esn_params into a hdf5 file.
       The h5py file has to be init. with ReadModel (saving the fix parameters) before calling this function!
       
       INPUT:
          filepath    - path to which the hdf5 file of the ESN study is saved to
          randomSeed  - ESN RNG seed
          isetting    - ESN grid search setting identifier
          study_dict  - dictionary specifying the studied parameter configuration
          y_pred_test - reseroir outputs of testing phase
          y_pred_val  - reseroir outputs of validation phase
          loss_dict   - losses of ESN training, testing & validation phase
          f           - opened hdf5.File where study will be saved to. If None, hdf5 file in filepath will be opened
       '''
           
    #HDF5-Structure:
        #- seed1
        #   - setting1
        #       - study_dict
        #       - y_pred
        #       - mse
        #       - randomSeed
        #
        #   - setting2
        #       ...  
        #   - ...
        #- seed2
        #   - setting1
        #       ...
        #   - setting2
        #   - ...
        #- ...
        
    to_close = False
    logger.info('Saving study to {0}'.format(filepath))

    if f is None:
        assert os.path.isfile(filepath), "Error: The file {0} does not exist. Did you initialize the file with util.create_hdf5_groups ?".format(filepath)
        f = h5py.File(filepath, 'a')
        to_close = True

    if 'Study' in f:
        G_study = f['Study']
    else:
        G_study = f.create_group('Study')


    if f'{randomSeed}' in G_study:
        G_seed = G_study[f'{randomSeed}']
    else:
        G_seed = G_study.create_group(f'{randomSeed}')

    if f'{isetting}' in G_seed:
        G_setting = G_seed.get(f'{isetting}')
    else:
        G_setting = G_seed.create_group(f'{isetting}')


    for param in study_dict.keys():
        G_setting.attrs[param] = study_dict[param]  
    
    #----------------------------------
    #  Save random seed
    #----------------------------------    
    G_setting.attrs["randomSeed"] = randomSeed
    
    #----------------------------------
    #  Datasets
    #----------------------------------    
    if 'y_pred_test' in G_setting:
        del G_setting['y_pred_test'] 
    
    G_setting.create_dataset('y_pred_test', data = y_pred_test, compression = 'gzip', compression_opts = 9)

    if 'y_pred_val' in G_setting:
        del G_setting['y_pred_val']
    
    G_setting.create_dataset('y_pred_val', data = y_pred_val, compression = 'gzip', compression_opts = 9)

    for loss_label,loss_value in loss_dict.items():
        if loss_label in G_setting:
            del G_setting[loss_label]
        
        G_setting.create_dataset(loss_label, data=loss_value, compression = 'gzip', compression_opts = 9)

    if to_close:
        f.close()

###########################################################################################################

#                            READING AN ESN STUDY

###########################################################################################################
#--------------------------------------------------------------------------
# FH added 28.02.2021
#--------------------------------------------------------------------------
def create_study_config_list(study_tuple: Union[list,tuple], study_dicts: dict) -> np.ndarray:
    ''' Computes an array, which gives the parameter configuration/setting for the corresponding study.
        
        INPUT: 
            study_tuple      - list/tuple of strings specifying which parameters were studied. 
                               E.g. when reservoir size and density are studied  study_tuple = ('n_reservoir', 'reservoirDensity')
            study_dicts      - dictionary specifying the study parameter configuration

        RETURN:
            - config - list indicating the parameter setting for given study
        '''

    nparam = len(study_tuple)      #no. different parameters that are studied
    nsetting = len(study_dicts)      #no. studies/ parameter settings that were conducted
    config = []   #np.empty([nsetting,nparam]) # FH 22.08.22: if HP is an array, config must be list

    for ii in range(nsetting):
        config_dict  =study_dicts[ii]

        config_param = []
        for pp in range(nparam):
            key = study_tuple[pp]
            config_param.append(config_dict[key])
            
        config.append(config_param)

    return config

#--------------------------------------------------------------------------
def read_study(filepath: str, 
               iseeds: Union[list,int],
               study_tuple: Union[list,tuple],
               read_pred:bool=False) -> Tuple[dict, np.ndarray, np.ndarray,list]:
    '''Imports the results of the ESN study. 
        
        INPUT:
            filepath         - path to which the hdf5 file of the ESN study was saved to
            iseeds           - list or int of ESN RNG seed identifier
            study_tuple      - list/tuple of strings specifying which parameters were studied. E.g. when reservoir size and density are studied: study_tuple = ['n_reservoir', 'reservoirDensity']
            read_pred        - boolean specifying whether to import ESN predictions y
            
        RETURN:
            loss_dict   - dict containing lossed from specified seeds 
            y_pred_test - reseroir outputs of testing phase, for each study parameter setting of the study 
            y_pred_val  - reseroir outputs of validation phase, for each study parameter setting of the study            
            config      - array indicating the parameter setting for given study
    '''

    assert os.path.isfile(filepath),"Error: File {0} not found.".format(filepath)

    if read_pred:
        logger.debug("Reading reservoir outputs.")

    if isinstance(iseeds,int):
        iseeds = [iseeds,]

    study_dicts  = list()
    loss_dict    = dict()
    y_pred_test, y_pred_val = list(),list()

    with h5py.File(filepath,'r') as f:
        
        G_study = f['Study']
        loss_label = list(f['Data'].attrs['loss_func'])

        # Initialize loss_dict
        #---------------------------
        loss_dict['mse_train'] = list()
        for label in loss_label:
            loss_dict[label+'_test'] = list()
            loss_dict[label+'_val'] = list()
        
        # Read loss & predictions
        #---------------------------
        for ii,iseed in enumerate(iseeds):
            G_seed = G_study[f'{iseed}']
            nsetting = len(G_seed.keys())
        
            Y_pred_test, Y_pred_val = [], []
            loss_dict_seed = {}
            loss_dict_seed['mse_train'] = list()
            for label in loss_label:
                loss_dict_seed[label+'_test'] = list()
                loss_dict_seed[label+'_val'] = list()

            for isetting in range(nsetting):
                G_setting = G_seed[f'{isetting}']
                
                # read study configuration
                if ii == 0:
                    study_dict = {}
                    for name in study_tuple:
                        study_dict[name] = G_setting.attrs[name]
                    study_dicts.append(study_dict)
        
                # read predictions
                if read_pred:
                    Y_pred_test.append(np.array(G_setting['y_pred_test']))
                    Y_pred_val.append(np.array(G_setting['y_pred_val']))

                # read losses
                loss_dict_seed['mse_train'].append(np.array(G_setting['mse_train']))
                for label in loss_label:
                    loss_dict_seed[label+'_test'].append(np.array(G_setting[label+'_test']))
                    loss_dict_seed[label+'_val'].append(np.array(G_setting[label+'_val']))
                
            # collect (single seed) predictions
            y_pred_test.append(np.array(Y_pred_test))
            y_pred_val.append(np.array(Y_pred_val))

            # collect (single seed) losses
            loss_dict['mse_train'].append(loss_dict_seed['mse_train'])
            for label in loss_label:
                loss_dict[label+'_test'].append(loss_dict_seed[label+'_test'])
                loss_dict[label+'_val'].append(loss_dict_seed[label+'_val'])

        # Reconstruct study configurations
        #----------------------------------
        config = create_study_config_list(study_tuple,study_dicts)

        # Convert to np.ndarray
        #----------------------------------     
        for label,loss in loss_dict.items():
            loss_dict[label] = np.array(loss)

        return loss_dict, np.array(y_pred_test), np.array(y_pred_val), config

#--------------------------------------------------------------------------
def read_loss(filepath: str, 
              iseeds: list,
              study_tuple: Union[list,tuple]) -> Tuple[dict, list]:
    '''Imports the MSEs of the ESN study. 
        
        INPUT:
            filepath         - path to which the hdf5 file of the ESN study was saved to
            iseeds           - list or int of ESN RNG seed identifier
            study_tuple      - list/tuple of strings specifying which parameters were studied. E.g. when reservoir size and density are studied: study_tuple = ['n_reservoir', 'reservoirDensity']

        RETURN:
            loss_dict   - dict containing lossed from specified seeds 
            config      - array indicating the parameter setting for given study
    '''

    assert os.path.isfile(filepath),"Error: File {0} not found.".format(filepath)

    if isinstance(iseeds,int):
        iseeds = [iseeds,]
    
    loss_dict = {}
    study_dicts = list()

    with h5py.File(filepath,'r') as f:

        G_study = f['Study']        
        loss_label = list(f['Data'].attrs['loss_func'])

        # Initialize loss_dict
        #---------------------------
        loss_dict['mse_train'] = list()
        for label in loss_label:
            loss_dict[label+'_test'] = list()
            loss_dict[label+'_val'] = list()
        
        # Read loss & predictions
        #---------------------------
        for ii,iseed in enumerate(iseeds):
            G_seed = G_study[f'{iseed}']
            nsetting = len(G_seed.keys())
        
            loss_dict_seed = {}
            loss_dict_seed['mse_train'] = list()
            for label in loss_label:
                loss_dict_seed[label+'_test'] = list()
                loss_dict_seed[label+'_val'] = list()

            for isetting in range(nsetting):
                G_setting = G_seed[f'{isetting}']
                
                # read study configuration
                if ii == 0:
                    study_dict = {}
                    for name in study_tuple:
                        study_dict[name] = G_setting.attrs[name]
                    study_dicts.append(study_dict)
        
                # read losses
                loss_dict_seed['mse_train'].append(np.array(G_setting['mse_train']))
                for label in loss_label:
                    loss_dict_seed[label+'_test'].append(np.array(G_setting[label+'_test']))
                    loss_dict_seed[label+'_val'].append(np.array(G_setting[label+'_val']))
                
            # collect (single seed) losses
            loss_dict['mse_train'].append(loss_dict_seed['mse_train'])
            for label in loss_label:
                loss_dict[label+'_test'].append(loss_dict_seed[label+'_test'])
                loss_dict[label+'_val'].append(loss_dict_seed[label+'_val'])

    # Reconstruct study configurations
    #----------------------------------         
    config = create_study_config_list(study_tuple, study_dicts)

    # Convert to np.ndarray
     #----------------------------------     
    for label,loss in loss_dict.items():
        loss_dict[label] = np.array(loss)

    return loss_dict, config

#--------------------------------------------------------------------------
def read_esn_output(filepath: str, 
                    iseeds: list,
                    study_tuple: Union[list,tuple]) -> Tuple[np.ndarray,np.ndarray, list]:
    '''Imports the outputs of the ESN study. 
        
        INPUT:
            filepath         - path to which the hdf5 file of the ESN study was saved to
            iseeds           - list or int of ESN RNG seed identifier
            study_tuple      - list/tuple of strings specifying which parameters were studied. E.g. when reservoir size and density are studied: study_tuple = ['n_reservoir', 'reservoirDensity']

        RETURN:
            y_pred_test - reseroir outputs of testing phase, for each study parameter setting of the study 
            y_pred_val  - reseroir outputs of validation phase, for each study parameter setting of the study          
            config      - array indicating the parameter setting for given study
    '''

    assert os.path.isfile(filepath),"Error: File {0} not found.".format(filepath)

    if isinstance(iseeds,int):
        iseeds = [iseeds,]

    y_pred_test, y_pred_val, study_dicts = list(), list(), list()
    with h5py.File(filepath,'r') as f:

        G_study = f['Study']  
        for ii,iseed in enumerate(iseeds):
            G_seed = G_study[f'{iseed}']
            nsetting = len(G_seed.keys())
        
            Y_pred_test, Y_pred_val  = [], []

            for isetting in range(nsetting):
                G_setting = G_seed[f'{isetting}']

                if ii == 0:
                    study_dict = {}
                    for name in study_tuple:
                        study_dict[name] = G_setting.attrs[name]
                    study_dicts.append(study_dict)
        
                Y_pred_test.append(np.array(G_setting['y_pred_test']))
                Y_pred_val.append(np.array(G_setting['y_pred_val']))

            # collect (single seed) predictions
            y_pred_test.append(np.array(Y_pred_test))
            y_pred_val.append(np.array(Y_pred_val))

    # Reconstruct study configurations
    #----------------------------------  
    config = create_study_config_list(study_tuple, study_dicts)

    return np.array(y_pred_test), np.array(y_pred_val), config
    
###########################################################################################################

#                            EXPERIMENTAL

###########################################################################################################
import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
clrs = prop_cycle.by_key()['color']

def plot_activation_arg_distribution(esn,
                                    style_dict,
                                    bins_input: Union[int,torch.Tensor] = 40, 
                                    bins_res: Union[int,torch.Tensor] = 40, 
                                    bins_fb: Union[int,torch.Tensor] = 40,):

    """
    Plots activation argument distribution from compute_activation_arg_distribution method in core.py
    INPUT:
        esn         - ESN object
        style_dict  - dict specifying plot parameters
        bins_input  - bins to use for computing the distribution of Win@u (inputs)
        bins_res    - bins to use for computing the distribution of Wres@x (reservoir states)
        bins_fb     - bins to use for computing the distribution of Wfb@y (feedbacks)
        
    RETURN:
        fig - figure 
        axs - axes
    """
    # Read Style
    #-----------------
    base_size1,base_size2 = style_dict["base_size1"], style_dict["base_size2"]
    
    try:
        fs_label = style_dict["fs_label"]
    except KeyError:
        fs_label = 15

    try:
        fs_tick = style_dict["fs_tick"]
    except KeyError:
        fs_tick = 12

    try:
        lw = style_dict["lw"]
    except KeyError:
        lw = 12
        
    act = lambda x: np.tanh(x)
    phase = ['total', 'input', 'reservoir', 'feedback']
    xlabels = [r'$W^{\rm in}u+W^{\rm r}x+W^{\rm fb}y$', r'$W^{\rm in}u$', r'$W^{\rm r}x$', r'$W^{\rm fb}y$']
    xlabels_tanh = [r'$\tanh($'+label+r')' for label in xlabels]

    # Plot
    #-----------------
    fig,axs = plt.subplots(figsize=(base_size1*4,base_size2),ncols=4,sharey=True,nrows=2)

    hists_train, bins_train = esn.compute_activation_arg_distribution(40,40,40,phase='train')
    for ii,(h,b) in enumerate(zip(hists_train,bins_train)):
        axs[0,ii].plot(b[:-1],h,label="training phase",linewidth=lw)
        axs[1,ii].plot(act(b[:-1]),h,label="training phase",linewidth=lw)
        
    hists_test, bins_test = esn.compute_activation_arg_distribution(40,40,40,phase='test')
    for ii,(h,b) in enumerate(zip(hists_test,bins_test)):
        axs[0,ii].plot(b[:-1],h,linestyle='dashed',label="testing phase",linewidth=lw)
        axs[1,ii].plot(act(b[:-1]),h,linestyle='dashed',label="testing phase",linewidth=lw)


    for iiax in axs:
        for iax in iiax:
            iax.tick_params(labelsize=fs_tick)
    for ii,iax in enumerate(axs[0]):
        iax.set_title(phase[ii],fontsize=fs_label)
        iax.set_xlim([-3,3])
        iax.set_xlabel(xlabels[ii],fontsize=fs_label)
    for ii,iax in enumerate(axs[1]):
        iax.set_xlim([-1.5,1.5])
        iax.set_xlabel(xlabels_tanh[ii],fontsize=fs_label)
        
        
    twin_x = axs[1,0].twinx()
    xx = np.linspace(-3,3,100)
    twin_x.plot(act(xx),xx,color="black",label=r"$\tanh(x)$")
    twin_x.set_yticks([])
    twin_x.legend(fontsize=fs_label)
    axs[0,0].legend(loc="lower left",bbox_to_anchor=(0,.68),fontsize=fs_label)
    plt.subplots_adjust(wspace=0.15,hspace=0.3)

    return fig, axs

#--------------------------------------------------------------------------
def plot_esn_predictions(esn,y_pred,style_dict,modes=None,phase='test'):
    """
    Plots ESN prediction from specified phase. 
    INPUT:
        esn        - ESN object
        y_pred     - ESN prediction  
        style_dict - dict specifying plot parameters
        modes      - list specifying which modes of y_pred will be plotted
        phase      - str indicating for which phase the distributions should be computed: 'train', 'test', 'validation'
    
    RETURN:
        fig - figure 
        axs - axes
    """
    
    # Read Style 
    #-----------------
    base_size1,base_size2 = style_dict["base_size1"], style_dict["base_size2"]
    ylim = style_dict["ylim"]

    try:
        fs_label = style_dict["fs_label"]
    except KeyError:
        fs_label = 15

    try:
        fs_tick = style_dict["fs_tick"]
    except KeyError:
        fs_tick = 12

    try:
        lw = style_dict["lw"]
    except KeyError:
        lw = 12

    try:
        ylabels = style_dict["ylabels"]
    except KeyError:
        ylabels = [r'$a_{'+f'{ii+1}'+ r'}$'for ii in modes]

    if modes is None:
        modes = range(esn.n_output)
        
    
    if phase == 'train':
        y_gt = esn.y_train[self.transientTime:]
    elif phase == 'test':
        y_gt = esn.y_test
    elif phase == 'validation':
        y_gt = esn.y_val
    
    nt = y_gt.shape[0]
        
    # Plot
    #-----------------
    fig,axs = plt.subplots(figsize = (base_size1,base_size2*len(modes)), nrows = len(modes), sharex = True)

    for ii,imode in enumerate(modes):
        axs[ii].plot(range(nt),
                     y_gt[:,imode], 
                     label = 'GT',
                     color = clrs[0],
                     linewidth = lw)
        axs[ii].plot(range(nt),
                     y_pred[:,imode], 
                     label = 'ESN',linestyle="dashed",
                     color = clrs[1],
                     linewidth = lw)


        axs[ii].set_ylabel(ylabels[ii], fontsize = fs_label, rotation = 0, labelpad = 17)
        axs[ii].set_ylim(ylim)

    # tidy up
    for iax in axs:
        iax.tick_params(axis="x", labelsize = fs_tick)
        iax.tick_params(axis="y", labelsize = fs_tick)

    axs[-1].set_xlabel('time step',fontsize = fs_label);

    return fig,axs

#--------------------------------------------------------------------------