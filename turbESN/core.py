#turbESN
from ._modes import (_DTYPE, _DEVICE, _ESN_MODES, _WEIGTH_GENERATION, _EXTENDED_STATE_STYLES, _LOGGING_FORMAT, _FIT_METHODS)

import torch
import numpy as np
import sys
import os
import h5py 
import json
from typing import Union, Tuple, List
import logging

# Read hyperparameter.json
import importlib.resources as pkg_resources
with pkg_resources.path(__package__,'hyperparameters.json') as hp_dict_path:
    with open(hp_dict_path,'r') as f:
        HP_dict = json.load(f)  


class ESN:

    _ID_COUNTER = 0

    def __init__(self, 
                randomSeed: int,
                esn_start: int, 
                esn_end: int,
                trainingLength: int, 
                testingLength: int,
                validationLength: int,
                data_timesteps: int,
                n_input: int,
                n_output: int,
                n_reservoir: int = 256,
                leakingRate: float = 1.0, 
                spectralRadius: float = 0.95,
                reservoirDensity: float = 0.2,
                regressionParameter: float = 5e-2,
                bias_in: float = 1.0,
                bias_out: float = 1.0,
                outputInputScaling: float = 1.0,
                inputScaling: float = 1.0, 
                inputDensity: float = 1.0, 
                noiseLevel_in: float = 0.0,
                noiseLevel_out: float = 0.0,
                mode: str = "auto",
                weightGeneration: str = "uniform",
                extendedStateStyle: str = "default",
                transientTime: int = 50,
                use_feedback: bool = False,
                feedbackScaling: float = 1,
                fit_method: str = "tikhonov",
                verbose: bool = False):


        self.device = _DEVICE

        if verbose:
            self.logging_level = logging.DEBUG
        else:
            self.logging_level = logging.WARNING

        logging.basicConfig(format=_LOGGING_FORMAT, level= self.logging_level)
        
        assert mode in _ESN_MODES,'Error: unkown mode {0}. Choices {1}'.format(mode, _ESN_MODES)
        self.mode = mode                                             # prediction mode (auto - autonmous prediction, teacher - teacher forced)
        logging.info('ESN mode is ' + self.mode + '\n')

        assert extendedStateStyle in _EXTENDED_STATE_STYLES,'Error: unkown extended state style {0}. Choices {1}'.format(extendedStateStyle, _EXTENDED_STATE_STYLES)
        self.extendedStateStyle = extendedStateStyle        
        
        ESN._ID_COUNTER +=1
        self.id = ESN._ID_COUNTER                                    #reservoir instance ID. Used for logging to keep track of multiple processes.

        self.randomSeed = randomSeed                                 #RNG seed for random reservoir initalization
        self.esn_start = esn_start                                   #Index of the original data, at which the training output y_train will begin. 
                                                                     #Note that the training input u_train will therefore begin at the index esn_start-1. Therefore esn_start can not be 0!
        self.esn_end = esn_end                                       #Index of the original data, at which the testing/validation output y_test will end.
        self.trainingLength = trainingLength                         #no. time steps for the training data set
        self.testingLength = testingLength                           #no. time stes for the testing data set
        self.validationLength = validationLength                     #no. time stes for the validation data set
        self.data_timesteps = data_timesteps                         #no. time steps the orignal data has
        self.esn_timesteps = trainingLength + testingLength          #no. total resulting time steps for the esn 
        self.n_input = int(n_input)                                  #input data dimensions
        self.n_output = int(n_output)                                #output data dimensions
        self.n_reservoir = int(n_reservoir)                          #dimensions of reservoir state and W with shape (n_reservoir, n_reservoir)
        
        if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            self.xrows = int(1+n_reservoir+n_input)                  #dim of extended reservoir state: (xrows,timesteps). xrows = bias + n_reservoir + n_input
        else:
            self.xrows = int(1+2*n_reservoir)                        #dim of extended reservoir state: (xrows,timesteps). xrows = bias + n_reservoir + n_reservoir

        if isinstance(leakingRate,np.ndarray):                       #allow for neuron specific LR: see e.g Tanaka et al. Phys. Rev. Res. 4 (2022)
            leakingRate = torch.as_tensor(leakingRate,dtype=_DTYPE).reshape(self.n_reservoir,1)    
        elif isinstance(leakingRate,torch.Tensor):
            leakingRate = leakingRate.to(_DTYPE).reshape(self.n_reservoir,1)

        self.leakingRate = leakingRate                               #factor controlling the leaky integrator formulation (1 -> fully nonlinear, 0 -> fully linear)
        self.spectralRadius = spectralRadius                         #maximum absolute eigenvalue of Wres
        self.reservoirDensity = reservoirDensity                     #fraction of non-zero elements of Wres
        self.regressionParameter = regressionParameter               #ridge regression/ penalty parameter of ridge regression
        self.bias_in = bias_in                                       #input bias in the input mapping: Win*[1;u]
        self.bias_out = bias_out                                     #output bias in the final output mapping:  y = Wout*[outputbias; outputInputScaling*u; s]
        self.outputInputScaling = outputInputScaling                 #factor by which the input data should be scaled by in the final output mapping: y = Wout*[outputbias; outputInputScaling*u; s]
        self.inputScaling = inputScaling                             #scaling of the columns of the input matrix Win
        self.inputDensity = inputDensity                             #fraction of non-zero elements of Win
        self.noiseLevel_in = noiseLevel_in                           #amplitude of the gaussian noise term inside the activation function
        self.noiseLevel_out = noiseLevel_out                         #amplitude of the gaussian noise term outside the activation function

        assert weightGeneration in _WEIGTH_GENERATION,'Error: unknown weightGeneration {0}. Choices {1}'.format(weightGeneration, _WEIGTH_GENERATION)
        self.weightGeneration = weightGeneration                     #method the random weights Win, W should be initialized
        self.transientTime = transientTime                           #washout length for reservoir states
        self.use_feedback = use_feedback                             #if True, the reservoir uses feedback weights
        self.feedbackScaling = feedbackScaling                       #scaling of the columns of the feedback matrix Wfb

        assert fit_method in _FIT_METHODS, "Error: unknown fit_method {0}. Choices {1}".format(fit_method, _FIT_METHODS)
        self.fit_method =  fit_method


        # Init to None
        self.y_train = None           
        self.u_train = None
        self.y_test  = None
        self.u_test  = None
        self.y_val   = None
        self.u_val   = None
        self.u_pre_val = None
        self.y_pre_val = None

        self.test_init_input = None
        self.val_init_input  = None

        self.x_train = None
        self.x_test  = None
        self.x_val   = None

        self.Win  = None
        self.Wout = None
        self.Wfb  = None
        self.Wres = None

        self.loss_func = None
#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  ESN IMPLEMENTATION
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def createInputMatrix(self):
        '''
        Random initialization of the input weights. The weights are drawn from U[-0.5,0.5] or N[0,1] and subsequently scaled by inputScaling.
        '''
        
        logging.debug('Building input matrix')

        if self.weightGeneration == 'normal':
            self.Win = torch.randn(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE)
        else:
            self.Win = torch.rand(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE) - 0.5


        if self.inputDensity >= 1e-6:
            _mask = torch.rand(self.n_reservoir, 1 + self.n_input, dtype = _DTYPE, device = self.device) > self.inputDensity
            self.Win[_mask] = 0.0
        else:
            self.Win = torch.zeros(self.n_reservoir, 1 + self.n_input, device = self.device, dtype = _DTYPE)


        # Input Scaling
        #----------------
        if self.inputScaling is None:
            self.inputScaling = 1
        
        if np.isscalar(self.inputScaling):
            _inputScaling = torch.ones(self.n_input, device = self.device, dtype = _DTYPE) * self.inputScaling 
        elif len(self.inputScaling) != self.n_input:
            logging.critical(' inputScaling dimension ({0}) does not match input dimension ({1})\n'.format(len(self.inputScaling), self.n_input))
            raise RuntimeError
        else:
            _inputScaling = self.inputScaling

        _inputScaling = torch.vstack((torch.tensor(1, device = self.device, dtype = _DTYPE), _inputScaling.reshape(-1,1))).flatten()

        self.Win *= _inputScaling.reshape(1,1+self.n_input)
        
#--------------------------------------------------------------------------
    def createReservoirMatrix(self):
        '''
        Random initialization of reservoir weights. The weights are drawn from U[-0.5,0.5]. The reservoir density is set.
        Finally, the largest absolute eigenvalue is computed, by which Wres is normalized. Then Wres is scaled by the spectral radius.
        '''
            
        logging.debug('Building reservoir matrix')

        if self.weightGeneration == 'normal':
            self.Wres = torch.as_tensor(torch.randn(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) , dtype = _DTYPE, device = self.device)
        else:
            self.Wres = torch.as_tensor(torch.rand(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) - 0.5, dtype = _DTYPE, device = self.device)
 
        #prevent all-zero eigenvals 
        if self.reservoirDensity >= 1e-6 and self.spectralRadius != 0.0:

            _mask = torch.rand(self.n_reservoir, self.n_reservoir, dtype = _DTYPE, device = self.device) > self.reservoirDensity
        
            self.Wres[_mask] = 0.0
            _eig_norm = torch.abs(torch.linalg.eigvals(self.Wres))
            self.Wres *= self.spectralRadius / torch.max(_eig_norm)
        else:
            self.Wres = torch.zeros((self.n_reservoir, self.n_reservoir), dtype = _DTYPE, device = self.device)

#--------------------------------------------------------------------------
    def createFeebackMatrix(self):
        '''
        Random initialization of feedback weights. The weights are drawn from U[-0.5,0.5] or N[0,1] and subsequently scaled by feedbackScaling.
        '''
        logging.debug('Building feedback matrix')

        if self.weightGeneration == 'normal':
            self.Wfb = torch.randn(self.n_reservoir, self.n_output, device = self.device, dtype = _DTYPE)
        else:
            self.Wfb = torch.rand(self.n_reservoir, self.n_output, device = self.device, dtype = _DTYPE) - 0.5


        # Feedback Scaling
        #------------------
        if self.feedbackScaling is None:
            self.feedbackScaling = 1

        if np.isscalar(self.feedbackScaling):
            _feedbackScaling = torch.ones(self.n_output, device = self.device, dtype = _DTYPE) * self.feedbackScaling 

        elif len(self.feedbackScaling) != self.n_output:
            logging.critical(' feedbackScaling dimension ({0}) does not match output dimension ({1})\n'.format(len(self.feedbackScaling), self.n_output))
            raise RuntimeError
        else:
            _feedbackScaling = self.feedbackScaling
        
        self.Wfb *= _feedbackScaling.reshape(1,self.n_output)
        
#--------------------------------------------------------------------------
    def createWeightMatrices(self):
        '''
        Reservoir initialization. The (fixed) weight matrices Win and Wres are created.
        '''
        self.createInputMatrix()
        self.createReservoirMatrix()

        if self.use_feedback:
            self.createFeebackMatrix()

#--------------------------------------------------------------------------
    def calculateLinearNetworkTransmissions(self, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ''' 
        Computes input and state contributions to the activation function.
        
        INPUT:
            u - current reservoir input
            x - last reservoir state
            y - last reservoir output
        RETURN: Win@u + Wres@x + Wfb@y
            
        '''
        
        uu = u.reshape(self.n_input, 1)
        feedback = 0

        if self.use_feedback:
            yy = y.reshape(self.n_output, 1)
            feedback = torch.matmul(self.Wfb,yy)
        
        return torch.matmul(self.Win, torch.vstack((torch.tensor(self.bias_in, device = self.device, dtype = _DTYPE), uu))) + torch.matmul(self.Wres, x) + feedback

#--------------------------------------------------------------------------
    def update(self, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor=None):
        '''
        Single update step of the reservoir state.

        INPUT:
            u - current reservoir input
            x - last reservoir state (will be updated here)
            y - last reservoir output
        '''

        transmission = self.calculateLinearNetworkTransmissions(u=u,x=x,y=y)

        x *= (1.0 - self.leakingRate)

        x += self.leakingRate * (torch.tanh( transmission 
                                             + self.noiseLevel_in  * (torch.rand((self.n_reservoir,1), device = self.device, dtype = _DTYPE) - 0.5)
                                           ) + self.noiseLevel_out * (torch.rand((self.n_reservoir,1), device = self.device, dtype = _DTYPE) - 0.5)
                                )

#--------------------------------------------------------------------------
    def propagate(self, u: torch.Tensor, x: torch.Tensor = None, transientTime: int = 0, y: torch.Tensor=None) -> torch.Tensor:
        ''' 
        Propagates the reservoir state x according to the reservoir dynamics. The number of time iterations is assumed from the 
        specified input u. The transien time specifies how many iterations are discarded for the reservoir washout. 
        These transient reservoir states will not be saved. 

        INPUT:
            u            - sequence of reservoir inputs
            x            - starting reservoir state
            transienTime - reservoir washout length
            y            - sequence of reservoir outputs (for feedback)

        RETURN:
            X - reservoir state matrix
        '''
        #TO DO: dirty fix for logger in prediction phase
        if transientTime != 0:
            logging.debug('Propagating states')


        if u is None:
            logging.critical('No reservoir input for propagation has been provided!\n')
            raise RuntimeError

        if self.use_feedback:
            if y is None:
                logging.critical(' Feedback: last output must be provided for state propagation \n')
                raise RuntimeError
            if y.shape[0] != u.shape[0]:
                logging.critical(' Feedback: provided output time dimension ({0}) does not match input time dimension ({1}) \n'.format(y.shape[0],u.shape[0]))
                raise RuntimeError            


        if u.dtype is not _DTYPE:
            u = torch.as_tensor(u, device = self.device, dtype = _DTYPE)

        #default intial reservoir state.
        if x is None:
            x = torch.zeros((self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        else:
            x = x.reshape(self.n_reservoir,1)

        input_timesteps = u.shape[0]

        # state matrix
        X = torch.zeros((self.xrows, input_timesteps - transientTime), device = self.device, dtype = _DTYPE)
        
        for t in range(input_timesteps):

            if self.use_feedback:
                self.update(u[t], x = x, y=y[t])
            else:
                self.update(u[t], x = x)
            
            if t >= transientTime:
                
                if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
                    X[:, t - transientTime] = torch.vstack(
                        (torch.tensor(self.bias_out, device = self.device, dtype = _DTYPE), 
                         self.outputInputScaling * u[t].reshape(self.n_input,1),
                          x))[:, 0]
                else:
                    X[:, t - transientTime] = torch.vstack(
                        (torch.tensor(self.bias_out, device = self.device, dtype = _DTYPE), 
                         x,
                         x**2))[:, 0]
        return X
#--------------------------------------------------------------------------
    def verify_echo_state_property(self, 
                                   u: torch.Tensor=None, 
                                   proximity_distance: float = 1e-3, 
                                   proximity_time: int = 20,
                                   y: torch.Tensor=None)->int:
        ''' Computes the convergence of two independent states: -1 and 1, when encountering the data u_train.
            INPUT:
                u                  - training input data (with which the reservoir is beeing forced)
                proximity_distance - proximity distance of the two initially independent states
                proximity_time     - no. time steps two initially independent states must be closer than proximity_distance
                y                  - training output (only used when feedback is used)

            RETURN:
                transientTime - no. time steps needed for two states -1 and 1, given the current reservoir and input u, to converge to a 'similar state'.
        '''

        if u is None:
            u = self.u_train
        if self.use_feedback and y is None:
            y = self.y_train
            
        input_timesteps = u.shape[0]

        x_init    =  torch.empty((2,self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        x_init[0] =  torch.ones((self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        x_init[1] = -torch.ones((self.n_reservoir, 1), device = self.device, dtype = _DTYPE)
        

        steps = 0
        for it in range(input_timesteps):
            
            #peak to peak distance
            ptp = x_init.max(dim = 0).values - x_init.min(dim = 0).values
            
            #states are close
            if torch.max(ptp) < proximity_distance:
                if steps >= proximity_time:
                    return it - proximity_time
                else: 
                    steps +=1
            
            #states are not close
            else:
                steps = 0

            #update reservoir states
            for ii in range(x_init.shape[0]):
                if self.use_feedback:
                    self.update(u[it], x_init[ii], y=y[it])
                else:
                    self.update(u[it], x_init[ii])
                    
        
        logging.warn('Reservoir states did not converge.')
        return torch.inf
#--------------------------------------------------------------------------
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        '''
        Reservoir output weights Wout is computed by L2 penalized linear regression.

        INPUT:
            X - reservoir state matrix
            y - target training output
        '''

        logging.debug('Fitting output matrix')
        assert X.shape[1] == y.shape[0], "Time dimension of X ({0}) does not match time dimension of y ({1}).\nDid you forget to exclude the transientTime of y? ".format(X.shape[1], y.shape[0])
        
        if self.fit_method == "tikhonov":
            I = torch.eye(self.xrows, device = self.device)
            I[0,0] = 0  #do not include bias term in regularization, see Lukosevicius et al. Cogn. Comp. (2021) p.2
            self.Wout = torch.matmul(torch.matmul(y.T,X.T), torch.inverse(X@X.T + self.regressionParameter*I))
        elif self.fit_method == "pinv":
            self.Wout = torch.matmul(y.T,torch.linalg.pinv(X))
        else:
            raise NotImplementedError("fit_method must be one of {0}".format(_FIT_METHODS))
#--------------------------------------------------------------------------
    def fetch_state(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Fetches reservoir state from extended reserovir state, depending on the extended reservoir state layout.

        INPUT:
            X - extended reservoir state

        RETURN:
            reservoir state

        '''
        if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            return X[int(1+self.n_input):].reshape(self.n_reservoir,1)
        else:
            return X[1:int(self.n_reservoir+1)].reshape(self.n_reservoir,1)

#--------------------------------------------------------------------------
    def predict(self, X: torch.Tensor, testingLength: int, init_input:torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Used in mode = auto. 

        Autonomous prediction mode of the reservoir. Starting from the last state of the training state matrix X and specified
        initial input init_input. If init_input not specified, rely on self.test_init_input. The reservoir feeds its last output back to the input layer.

        INPUT:
            X             - state matrix (from which the last state will be taken as starting point) 
            testingLength - number of iterations of the prediction phase
            init_input    - initial input to the ESN from which autonom. prediction will start. 

        RETURN:
            y_pred - reservoir output (predictions)
            x_pred - reservoir state matrix (prediction phase)
        '''

        logging.debug('Predicting output')

        if init_input is None:
            if self.test_init_input is None:
                logging.error('Error in predict: Initial prediction input is not defined! Returning default values for (y_pred, x_pred).')
                return torch.zeros((testingLength, self.n_output)), torch.zeros((self.xrows, testingLength))
            else:
                init_input = self.test_init_input


        y_pred = torch.zeros([self.n_output,testingLength], device = self.device, dtype = _DTYPE) 
        x_pred = torch.zeros([self.xrows, testingLength], device = self.device, dtype = _DTYPE)


        x = X[:,-1].reshape(self.xrows,)
        pred_input = init_input

        for it in range(testingLength):
            x_in = self.fetch_state(x)

            if self.use_feedback:
                y = (self.Wout@x).T.reshape(1,self.n_output)
            else:
                y = None

            x = self.propagate(pred_input.reshape(1,self.n_input),x = x_in, y=y)      #state at time it
            
            pred_output = self.Wout@x                               #reservoir output at time it    

            y_pred[:,it] = pred_output.reshape(self.n_output,)
            x_pred[:,it] = x.reshape(self.xrows,)
            
            pred_input = pred_output                             #new input is the current output
        
        return y_pred.T, x_pred

#--------------------------------------------------------------------------
    def teacherforce(self, X: torch.Tensor, testingLength: int, u: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Use in mode = teacher.

        Teacher forcing mode of the reservoir. Starting from the last state of the training state matrix X. Inputs are given by self.u_test.
        The reservoir receives an external input u_test[it] each time step. 

        INPUT:
            X             - state matrix (from which the last state will be taken as starting point) 
            testingLength - number of iterations of the prediction phase
            u             - teacher signal (if None, self.u_test is used)

        RETURN:
            y_pred - reservoir output (predictions)
            x_pred - reservoir state matrix (prediction phase)
        '''
        
        logging.debug('Predicting output')
            
        y_pred = torch.zeros((self.n_output,testingLength), device = self.device, dtype = _DTYPE) 
        x_pred = torch.zeros((self.xrows, testingLength), device = self.device, dtype = _DTYPE)
        x = X[:,-1].reshape(self.xrows,)

        if u is None:
            u = self.u_test

        #compute reservoir states
        for it in range(testingLength):
            x_in = self.fetch_state(x)
    
            if self.use_feedback:
                y = (self.Wout@x).T.reshape(1,self.n_output)
            else:
                y = None

            x = self.propagate(u[it].reshape(1,self.n_input),x=x_in, y=y)      #state at time it

            x_pred[:,it] = x.reshape(self.xrows,)
            

        #compute reservoir outputs
        y_pred = self.Wout@x_pred                                                     
    
        return y_pred.T, x_pred

#--------------------------------------------------------------------------
    def semiteacherforce(self, X: torch.Tensor, testingLength: int, index_list_auto: list, index_list_teacher: list, u_teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Use in mode: semi-teacher.

        Semi-teacher forcing mode of the reservoir. Starting from the last state of the training state matrix X. 
        Inputs are partially given by self.u_test and autonomous predictions.
        The reservoir receives an external input u_test[it] each time step.
        For now restricted to cases, where n_input = n_output & no feedback weights!

        INPUT:
            X                  - state matrix (from which the last state will be taken as starting point) 
            testingLength      - number of iterations of the prediction phase
            index_list_auto    - indices of the modes which are passed back as new input
            index_list_teacher - indices of the modes which are supplied by a teacher signal

        RETURN:
            y_pred - reservoir output (predictions)
            x_pred - reservoir state matrix (prediction phase)
        '''
        
        logging.debug('Predicting output')

        y_pred = torch.zeros([self.n_output,testingLength], device = self.device, dtype = _DTYPE) 
        x_pred = torch.zeros([self.xrows, testingLength], device = self.device, dtype = _DTYPE)
        x = X[:,-1].reshape(self.xrows,)


        def construct_input(index_list_auto, index_list_teacher, u_auto, u_teacher):
            '''Construct initial input from autonomous signal and teacher signal.'''

            u = torch.zeros(self.n_input)

            for aa in range(len(index_list_auto)):
                index = index_list_auto[aa]
                u[index] = u_auto[aa]
            for tt in range(len(index_list_teacher)):
                index = index_list_teacher[tt]
                u[index] = u_teacher[tt]

            return u

        pred_input = self.test_init_input

        for it in range(testingLength):

            u_auto = pred_input[index_list_auto]     # autonomous part of prediction: used as part of new input
            u_test = self.u_test[it]                 # teacher part of prediction: used as part of new input
            u_merged = construct_input(index_list_auto, index_list_teacher, u_auto, u_test)
        
            x_in = self.fetch_state(x)
            x = self.propagate(u_merged.reshape(1,self.n_input),x = x_in)      #state at time it

            pred_output = self.Wout@x  
            y_pred[:,it] = pred_output.reshape(self.n_output,)
            x_pred[:,it] = x.reshape(self.xrows,)

            pred_input = pred_output


        return y_pred.T, x_pred


#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  HYPERPARAMETER SETTERS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    
    #FH 01/02/2021: Added SetTrainingData & SetTestingData
    #--------------------------------------------------------------------------
    def SetTrainingData(self, u_train: torch.Tensor, y_train: torch.Tensor):
        
        assert u_train.shape[0] == y_train.shape[0],'Training input dimension ({0}) does not match training output time dimension ({1}).\n'.format(u_train.shape[0], y_train.shape[0])
        assert u_train.shape[1] == self.n_input,'Training input dimension ({0}) does not match ESN n_input ({1}).\n'.format(u_train.shape[1], self.n_input)
        assert y_train.shape[1] == self.n_output, 'Training output dimension ({0}) does not match ESN n_output ({1}).\n'.format(y_train.shape[1], self.n_output)
        
        self.u_train = u_train
        self.y_train = y_train
    
    #--------------------------------------------------------------------------
    def SetTestingData(self, y_test: torch.Tensor, test_init_input: torch.Tensor = None, u_test: torch.Tensor = None):       
        
        assert y_test.shape[1] == self.n_output,'Testing output dimension ({0}) does not match ESN n_output ({1}).\n'.format(y_test.shape[1], self.n_output)
    
        self.y_test = y_test
        self.u_test = u_test

        if self.mode == _ESN_MODES[0]:

            if self.y_train is None:
                assert test_init_input is not None, 'Initial testing input and self.y_train not specified.\n'

            if self.y_train is not None and test_init_input is None:
                #Initial input is last training input. Then first prediction aligns with the first entry of y_test
                logging.debug('Initial testing input not specified. Using last target training output.')
                self.test_init_input = self.y_train[-1:,:]    #initial input the trained ESN receives for the beginning of the testing phase
           
            elif test_init_input is not None:
                self.test_init_input = test_init_input

        elif self.mode == _ESN_MODES[1]:
            assert u_test is not None, 'Teacher mode requires non empty u_test!\n'

        if self.mode == _ESN_MODES[2]:

            if test_init_input is not None:
                self.test_init_input = test_init_input

                #TO DO: test_init_input has len n_input. The teacher part of it is not used (see self.semiteacherforce).

    #--------------------------------------------------------------------------
    # FH 30.03.2022: Added Validation Datset (auto mode!)
    def SetValidationData(self, 
                          y_val: torch.Tensor, 
                          u_val: torch.Tensor=None, 
                          val_init_input: torch.Tensor=None, 
                          u_pre_val: torch.Tensor=None,
                          y_pre_val: torch.Tensor=None):   

        assert y_val.shape[1] == self.n_output,'Validation output dimension ({0}) does not match ESN n_output ({1}).\n'.format(y_val.shape[1], self.n_output)
        
        if self.mode == _ESN_MODES[1]:
            assert u_val is not None, 'Teacher mode requires non empty u_val!\n'

        if u_pre_val is not None:
            assert u_pre_val.shape[1] == self.n_input, 'Validation state initialisation dimension ({0}) does not match ESN n_input ({1}).\n'.format(u_pre_val.shape[1], self.n_input)

        self.u_val = u_val
        self.y_val = y_val

        if u_pre_val is not None:
            self.u_pre_val = u_pre_val

            if self.use_feedback:
                assert y_pre_val is not None, 'When using feedback & pre validation state propagation (u_pre_val) a target output has to be provided!'
                self.y_pre_val = y_pre_val


        if self.mode == _ESN_MODES[0]:
            if val_init_input is None:
                #Initial input is last testing input. Then first prediction aligns with the first entry of y_val
                logging.debug('Initial validation input not specified. Using last target test output.')
                self.val_init_input = self.y_test[-1:,:]    #initial input the trained ESN receives for the beginning of the validation phase
            else:
                self.val_init_input = val_init_input

    #--------------------------------------------------------------------------
    #FH 14/11/2021: Added logging 
    def SetLoggingLevel(self, logging_level):
    
        logging.warn(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
    
    #--------------------------------------------------------------------------
    def SetID(self, esn_id):
    
        logging.debug(f'Setting reservoir ID to {esn_id}')
        self.id = esn_id

    #--------------------------------------------------------------------------
    def SetNInputOutput(self,n_input: int, n_output: int = None, study_dict: dict = {}):
        
        logging.debug(f'Setting n_input {n_input}')

        self.n_input = n_input
        if n_output is not None:
            logging.debug(f'Setting n_output {n_output}')
            self.n_output = n_output
        else:
            self.n_output = self.n_input 
        
        #adjust xrows as according to changed n_input
        if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            self.xrows = int(1+self.n_reservoir+self.n_input)          
        
        study_dict['n_input'] = n_input
     #--------------------------------------------------------------------------
    def SetNReservoir(self,n_reservoir: int, study_dict: dict = {}):

        logging.debug(f'Setting n_reservoir {n_reservoir}')
        
        self.n_reservoir = n_reservoir

        #adjust xrows as according to changed n_reservoir
        if self.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            self.xrows = int(1+self.n_reservoir+self.n_input)           
        else:
            self.xrows = int(1+2*self.n_reservoir)

        study_dict['n_reservoir'] = n_reservoir

    #--------------------------------------------------------------------------
    def SetSpectralRadius(self, spectralRadius: float, study_dict: dict = {}):
        
        logging.debug(f'Setting spectralRadius {spectralRadius}')
        
        self.spectralRadius = spectralRadius
        study_dict['spectralRadius'] = spectralRadius
    #--------------------------------------------------------------------------
    def SetReservoirDensity(self, reservoirDensity: float, study_dict: dict = {}):
        
        logging.debug(f'Setting reservoirDensity {reservoirDensity}')
        self.reservoirDensity = reservoirDensity
        study_dict['reservoirDensity'] = reservoirDensity
    #--------------------------------------------------------------------------
    def SetLeakingRate(self, leakingRate: float, study_dict: dict = {}):
       
        logging.debug(f'Setting leakingRate {leakingRate}')
        self.leakingRate = leakingRate
        study_dict['leakingRate'] = leakingRate
    #--------------------------------------------------------------------------
    def SetRegressionParameter(self, regressionParameter: float, study_dict: dict = {}):
        
        logging.debug(f'Setting regressionParameter {regressionParameter}')
        self.regressionParameter = regressionParameter
        study_dict['regressionParameter'] = regressionParameter
    #--------------------------------------------------------------------------
    def SetBiasIn(self, bias_in: float, study_dict: dict = {}):

        logging.debug(f'Setting bias_in {bias_in}')
        self.bias_in = bias_in
        study_dict['bias_in'] = bias_in
    #--------------------------------------------------------------------------
    def SetBiasOut(self, bias_out: float, study_dict: dict = {}):

        logging.debug(f'Setting bias_out {bias_out}')
        self.bias_out = bias_out
        study_dict['bias_out'] = bias_out
    #--------------------------------------------------------------------------
    def SetOutputInputScaling(self, outputInputScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting outputInputScaling {outputInputScaling}')
        self.outputInputScaling = outputInputScaling
        study_dict['outputInputScaling'] = outputInputScaling
    #--------------------------------------------------------------------------
    def SetInputScaling(self, inputScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting inputScaling {inputScaling}')
        self.inputScaling = inputScaling
        study_dict['inputScaling'] = inputScaling
    #--------------------------------------------------------------------------
    def SetFeedbackScaling(self, feedbackScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting feedbackScaling {feedbackScaling}')
        self.feedbackScaling = feedbackScaling
        study_dict['feedbackScaling'] = feedbackScaling
    
    #--------------------------------------------------------------------------
    def SetInputDensity(self, inputDensity: float, study_dict: dict = {}):
        
        logging.debug(f'Setting inputDensity {inputDensity}')
        self.inputDensity = inputDensity
        study_dict['inputDensity'] = inputDensity
    #--------------------------------------------------------------------------
    def SetNoiseLevelIn(self, noiseLevel_in: float, study_dict: dict = {}):
        
        logging.debug(f'Setting noiseLevel_in {noiseLevel_in}')
        self.noiseLevel_in = noiseLevel_in
        study_dict["noiseLevel_in"] = noiseLevel_in
    #--------------------------------------------------------------------------
    def SetNoiseLevelOut(self, noiseLevel_out: float, study_dict: dict = {}):
        
        logging.debug(f'Setting noiseLevel_out {noiseLeve_out}')
        self.noiseLevel_out = noiseLevel_out
        study_dict["noiseLevel_out"] = noiseLevel_out

    #--------------------------------------------------------------------------
    def SetTransientTime(self, transientTime: int, study_dict: dict = {}):
        
        logging.debug(f'Setting transientTime {transientTime}')
        self.transientTime = transientTime
        study_dict["transientTime"] = transientTime

    #--------------------------------------------------------------------------
    def SetDataScaling(self, dataScaling: float, study_dict: dict = {}):
        
        logging.debug(f'Setting dataScaling {dataScaling}')
        self.u_train *=dataScaling
        self.y_train *=dataScaling
        if self.u_test is not None:
            self.u_test  *=dataScaling
        self.y_test  *=dataScaling
        study_dict['dataScaling'] = dataScaling
    #--------------------------------------------------------------------------
    #FH 21/03/2021: added Seed Setter method
    def SetRandomSeed(self, randomSeed: int):
        
        logging.debug(f'Setting randomSeed {randomSeed}')
        self.randomSeed = randomSeed

        # TO DO: 
        # I guess this has no effect. The seed must be set in the script where the computations are done.
        if self.randomSeed is not None:
            torch.manual_seed(self.randomSeed)

    #--------------------------------------------------------------------------
    #FH 22/03/2021:  added automatic Setter method 
    def SetStudyParameters(self, config_istudy: Union[np.ndarray, torch.Tensor], study_tuple: tuple) -> dict:
        '''
        Sets the hyperparameters, specified in study_tuple, to values specified in config_istudy.
        Returns the changed setting in form of a dictionary. 

        Hyperparameters which are not included with their setter method, can not be studies by the grid/random search routine!
            
        INPUT:
            config_istudy  - array specifing values of the hyperparameters to which the ESN should be set
            study_tuple    - tuple of strings, specifying which hyperparameters are studied (order must be same as config_istudy)
        
        RETURN:
            study_dict - dict specifying the new hyperparameter setting of the ESN 
        '''
        
        assert len(study_tuple) > 0,'study_tuple is empty. Did you forget to specify the study HP?'

        study_dict = {}

        for iparam, parameter in enumerate(study_tuple):

            if parameter == "n_reservoir":
                self.SetNReservoir(config_istudy[iparam], study_dict)

            elif parameter == "spectralRadius":
                self.SetSpectralRadius(config_istudy[iparam], study_dict)

            elif parameter == "reservoirDensity":
                self.SetReservoirDensity(config_istudy[iparam], study_dict)

            elif parameter == "leakingRate":
                self.SetLeakingRate(config_istudy[iparam], study_dict)

            elif parameter == "regressionParameter":
                self.SetRegressionParameter(config_istudy[iparam], study_dict)
            
            elif parameter == "bias_in":
                self.SetBiasIn(config_istudy[iparam], study_dict)
            
            elif parameter == "bias_out":
                self.SetBiasOut(config_istudy[iparam], study_dict)

            elif parameter == "outputInputScaling":
                self.SetOutputInputScaling(config_istudy[iparam], study_dict)

            elif parameter == "inputScaling":
                self.SetInputScaling(config_istudy[iparam], study_dict)
            
            elif parameter == "feedbackScaling":
                self.SetFeedbackScaling(config_istudy[iparam], study_dict)
            
            elif parameter == "inputDensity":
                self.SetInputDensity(config_istudy[iparam], study_dict)

            elif parameter == "dataScaling":
                self.SetDataScaling(config_istudy[iparam], study_dict)

            elif parameter == "inputDensity":
                self.SetInputDensity(config_istudy[iparam], study_dict)
            
            elif parameter == "noiseLevel_in":
                self.SetNoiseLevelIn(config_istudy[iparam], study_dict)
            
            elif parameter == "noiseLevel_out":
                self.SetNoiseLevelOut(config_istudy[iparam], study_dict)
            
            elif parameter == "transientTime":
                self.SetTransientTime(config_istudy[iparam], study_dict)

            else:
                logging.critical('Specified parameter {0} unknown.'.format(parameter))
                raise KeyError
                
        return study_dict

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  MISC. IMPLEMENTATIONS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def to_torch(self):
        '''
        Moves the ESN object to python or torch types. Torch tensors are automatically shifted to device.
        This will stop numpy vs. torch clashes from happening.
        '''
        
        logging.debug('Moving class arrays to {0} with dtype {1}'.format(self.device, _DTYPE))

        allmembers = [m for m in dir(self) if m[:2] != '__' and m[-2:] != '__']
        ifaddmember = [not callable(getattr(self, name)) for name in allmembers ]
        
        #Cast from numpy to python/ torch types
        for ii,member in enumerate(allmembers):
            if ifaddmember[ii]:
                value = getattr(self,member)
                
                if isinstance(value, np.float32) or isinstance(value, np.float64):         #np.float   -> float
                    new_value = float(value)      

                elif isinstance(value, np.int32) or isinstance(value, np.int64):           #np.int   -> int
                    new_value = int(value) 
                
                elif isinstance(value, np.ndarray):         #array    -> tensor
                    new_value = torch.as_tensor(value, device = self.device, dtype = _DTYPE)
                
                elif isinstance(value, torch.Tensor):
                    if value.dtype != _DTYPE or value.device != self.device:
                        new_value = torch.as_tensor(value, dtype= _DTYPE, device = self.device)
                    else:
                        new_value = value
                else:
                    new_value = value
                    
                setattr(self, member, new_value)       
            
#--------------------------------------------------------------------------
    def set_device(self, device: str):
        ''' Set globael ESN device (cpu or cuda)'''
        
        logging.debug('Setting device: {0}'.format(device))
        self.device = device
        
#---------------------------------------------------------------------------
    def get_size(self):
        """
        Returns estimated size of ESN object in MB.

        RETURN:
            obj_size - estimated size of all torch.Tensors, np.ndarrays in MB
        """

        obj_size = 0
        
        for attr_str in dir(self):
            attr = getattr(self,attr_str)
            
            if isinstance(attr,torch.Tensor):
                obj_size += (attr.nelement()*attr.element_size())/1e6
            elif isinstance(attr,np.ndarray):
                obj_size += attr.nbytes/1e6
            else:
                continue

        return obj_size

#---------------------------------------------------------------------------
    def save(self,filepath: str, f = None):
        ''' Saves the reservoir parameters, training and validation/testing data from ESN class object into a hdf5 file
    
        INPUT:
            filepath   - path to which the hdf5 file of the ESN study is saved to
        '''
        
        to_close = False
        logging.warn('Saving ESN parameters to Hdf5 file {0}'.format(filepath))

        hdf5_attrs_save = (float,int,bool,list,str)
        hdf5_array_save = (np.ndarray,torch.Tensor)

        if f is None:
            f = h5py.File(filepath, 'w')
            to_close = True

        # Save Hyperparameters & Data sets
        G_hp = f.create_group('Hyperparameters')
        G_data = f.create_group('Data')
        for key in HP_dict.keys():
            try:
                attr = getattr(self,key)

                if HP_dict[key]["SAVE_TO_HDF5"]:
                    if isinstance(attr,hdf5_attrs_save):
                        G_hp.attrs[key]= attr
                    elif isinstance(attr,hdf5_array_save):
                        if key == "leakingRate":
                            G_hp.create_dataset(key,  data = attr, compression = 'gzip', compression_opts = 9)
                        else:
                            G_data.create_dataset(key, data = attr, compression = 'gzip', compression_opts = 9)
                    else:
                        G_hp.attrs[key]= False
            except AttributeError:
                pass

        # Save loss dict
        if self.loss_func is not None:
            G_data.attrs['loss_func'] = list(self.loss_func.keys())
        else:
            G_data.attrs['loss_func'] = list()

        if to_close:
            f.close()

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  DUNDER METHODS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def __repr__(self):

        repr = "--------------------------\n"
        repr += "ESN Setting:\n"
        repr += "--------------------------\n"
        if self.randomSeed is not None:
            repr += "randomSeed = {0}\n".format(self.randomSeed)
        else:
            repr += "randomSeed = None\n"
            
        repr += "n_input"+ " = {0:.0f}\n".format(self.n_input)
        repr += "n_output"+ " = {0:.0f}\n".format(self.n_output)
        
        repr += "n_reservoir" + " = {0:.0f}\n".format(self.n_reservoir)
        repr += "reservoir density = {0:.3f}\n".format(self.reservoirDensity)
        repr += "spectral radius = " + "{0:.3f}\n".format(self.spectralRadius)

        if np.isscalar(self.leakingRate):
            repr += "leaking rate = "+"{0:.3f}\n".format(self.leakingRate)
        else:
            repr += "leaking rate is array\n"
        
        repr += "regression parameter = "+"{0:.2e}\n".format(self.regressionParameter)
        
        repr += "training length = " + "{0:.0f}\n".format(self.trainingLength)
        repr += "testing length = " + "{0:.0f}\n".format(self.testingLength)
        repr += "validation length = " + "{0:.0f}\n".format(self.validationLength)
        repr += "esn start = " + "{0:.0f}\n".format(self.esn_start)
        repr += "esn end = " + "{0:.0f}\n".format(self.esn_end)
        repr += "data length = " + "{0:.0f}\n".format(self.data_timesteps)

        repr += "input bias = " + "{0:.0f}\n".format(self.bias_in)
        repr += "output bias = " + "{0:.0f}\n".format(self.bias_out)

        if np.isscalar(self.inputScaling):
            repr += "input scaling = " + "{0:.0f}\n".format(self.inputScaling)
        else:
            repr += "input scaling is array\n"

        if np.isscalar(self.feedbackScaling):
            repr += "feedback scaling = " + "{0:.0f}\n".format(self.feedbackScaling)
        else:
            repr += "feedback scaling is array\n"

        repr += "output input scaling = " + "{0:.0f}\n".format(self.outputInputScaling)
        
        repr += "noise level inside activation function = " + "{0:.2e}\n".format(self.noiseLevel_in)
        repr += "noise level outside activation function = " + "{0:.2e}\n".format(self.noiseLevel_out)

        repr += "transientTime = " + "{0}\n".format(self.transientTime)

        repr += "weight dist. = "+self.weightGeneration+"\n"
        repr += "use feedback weights: "+str(self.use_feedback)+"\n"
        repr += "ESN size [MB]: {0:.1f}".format(self.get_size())+"\n"
        
        return repr

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  CLASS METHODS
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    @classmethod
    def read(cls, filepath:str, f = None):
        '''
        Creates an ESN object from a saved hdf5 file.

        INPUT:
            filepath - path to the saved ESN 

        RETURN:
            esn - ESN object
        '''

        to_close = False
        logging.warn('Reading ESN parameters from Hdf5 file {0}'.format(filepath))

        esn = ESN.vanilla_esn()

        if f is None:
            try:
                f = h5py.File(filepath, 'r')
                to_close = True  
            except FileNotFoundError:
                logging.debug('Error: file {0} not found.'.format(filepath))
                return None

        # Read Hyperparameters
        G_hp = f.get('Hyperparameters')
        for key in G_hp.attrs.keys():
            attr = G_hp.attrs[key]
            setattr(esn,key,attr)

        # Inferred parameters
        if esn.extendedStateStyle == _EXTENDED_STATE_STYLES[0]:
            setattr(esn,'xrows',int(1+esn.n_reservoir+esn.n_input))
        else:
            setattr(esn,'xrows',int(1+2*esn.n_reservoir)   )

        # if leakingRate is array
        if "leakingRate" not in G_hp.attrs.keys():
            attr = torch.from_numpy(np.array(G_hp["leakingRate"])).to(_DTYPE)
            setattr(esn,"leakingRate",attr)

        # Read Data
        G_data = f.get('Data')
        for key in G_data.keys():
            attr = torch.from_numpy(np.array(G_data[key])).to(_DTYPE)
            setattr(esn,key,attr)


        # Read loss_func
        try:
            esn.loss_func = list(G_data.attrs['loss_func'])
        except KeyError:
            esn.loss_func = None
            
        # Set Training/Testing/Validation data
        if esn.u_train is not None and esn.y_train is not None:
            esn.SetTrainingData(u_train=esn.u_train, y_train=esn.y_train)
        if esn.y_test is not None:
            esn.SetTestingData(y_test=esn.y_test, u_test=esn.u_test)
        if esn.u_val is not None and esn.y_val is not None:
            esn.SetValidationData(y_val=esn.y_val, u_val=esn.u_val)     

        if to_close:
            f.close()
        
        return esn

 #--------------------------------------------------------------------------
    @classmethod
    def get_HP_info(cls):
        '''
            Prints all hyperparameter information.
        '''

        for key,val in HP_dict.items():
            print(f"{key} - " +val["INFO"])

 #--------------------------------------------------------------------------
    @classmethod
    def get_HP_dict(cls):
        '''
            Returns hyperparameter dict.
        '''
        return HP_dict

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  Standard ESNs
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    @classmethod
    def vanilla_esn(cls):

        esn = cls( randomSeed= 0,
                    esn_start=1, 
                    esn_end=200,
                    trainingLength=100, 
                    testingLength=50,
                    validationLength=50,
                    data_timesteps=200,
                    n_input=1,
                    n_output=1)

        return esn

#--------------------------------------------------------------------------     
#--------------------------------------------------------------------------
#  EXPERIMENTAL
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
    def compute_activation_arg_distribution(self, 
                                            bins_input: Union[int,torch.Tensor] = 40, 
                                            bins_res: Union[int,torch.Tensor] = 40, 
                                            bins_fb: Union[int,torch.Tensor] = 40,
                                            phase: str = 'test'):
        '''
        Inspect distribution of X_in, X_res, X_fb & X_total = X_in + X_res + X_fb, where X are the ESN weights times the input/reservoir state/feedback.

            tanh(X_in + X_res + X_fb)

        Input, reservoir state & feedback values are chosen according to specified ESN prediction phase.
        Might be of use for assment of activation saturation.
        Note that historgram of total arguments share the same bins as the input.

        INPUT:
            bins_input  - bins to use for computing the distribution of Win@u (inputs)
            bins_res    - bins to use for computing the distribution of Wres@x (reservoir states)
            bins_fb     - bins to use for computing the distribution of Wfb@y (feedbacks)
            phase       - str indicating for which phase the distributions should be computed: 'train', 'test', 'validation'
        RETURN:
            hist_total  - historam of Win@u + Wres@x
            hist_input  - histogram of Win@u
            hist_res    - histogram of Wres@x
            hist_fb     - histogram of Wfb@y

            bins_total  - bins of hist_total
            bins_input  - bins of hist_input
            bins_res    - bins of hist_res
            bins_fb     - bins of hist_fb

        '''

        if phase == 'train':
            x = self.x_train
            u = self.u_train[self.transientTime:]
            y = self.y_train[self.transientTime:]
        elif phase == 'test':
            x = self.x_test
            u = self.u_test
            y = self.y_test
        elif phase == 'validation':
            x = self.x_val
            u = self.u_val
            y = self.y_val
        else:
            raise NotImplementedError("Error: phase can only be train, test or validation")

        
        # Reservoir activation argument
        #-----------------------------
        arg_res = self.Wres@x[int(1+self.n_input):,:]
        hist_res, bins_res = np.histogram(arg_res.flatten(), bins = bins_res)
        hist_res   = hist_res/len(arg_res.flatten())


        # Input activation argument
        #-----------------------------
        if self.mode == _ESN_MODES[0]:
            inp = torch.cat((torch.ones((1,self.xrows), device = self.device, dtype = _DTYPE),self.Wout ),dim = 0)@x
            arg_input = self.Win@inp

        elif self.mode == _ESN_MODES[1]:
            inp = torch.cat((torch.ones((u.shape[0],1),device = self.device, dtype = _DTYPE),u),dim=1)
            arg_input = self.Win@inp.T

        hist_input, bins_input  = np.histogram(arg_input.flatten(), bins = bins_input)
        hist_input = hist_input/len(arg_input.flatten())
        

         # Feedback activation argument
        #-----------------------------
        if self.use_feedback:
            if self.mode == _ESN_MODES[0]:
                inp = torch.cat((torch.ones((1,self.xrows), device = self.device, dtype = _DTYPE),self.Wout ),dim = 0)@x
                arg_fb = self.Wfb@inp

            elif self.mode == _ESN_MODES[1]:
                arg_fb = self.Wfb@y
            
            hist_fb, bins_fb = np.histogram(arg_fb.flatten(), bins = bins_fb)
            hist_fb    = hist_fb/len(arg_fb.flatten())
        else:
            hist_fb = np.zeros(bins_fb)
            bins_fb = np.zeros(bins_fb+1)

        
        # Total activation argument (input+reservoir+feedback)
        #--------------------------------------------------------
        if not self.use_feedback:
            hist_total, bins_input  = np.histogram((arg_res+arg_input).flatten(), bins = bins_input)
            hist_total = hist_total/len((arg_res + arg_input).flatten())
        else:
             hist_total, bins_input  = np.histogram((arg_res+arg_input+arg_fb).flatten(), bins = bins_input)
             hist_total = hist_total/len((arg_res+arg_input+arg_fb).flatten())

        bins = [bins_input,bins_input,bins_res,bins_fb]
        hist = [hist_total, hist_input, hist_res, hist_fb]
        
        return hist,bins

#--------------------------------------------------------------------------
